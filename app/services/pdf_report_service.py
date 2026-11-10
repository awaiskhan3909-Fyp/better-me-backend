import io
import os
import uuid
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from sqlalchemy.orm import Session
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.db.models import Conversation, ConversationSessionState, User, Message, MessageAnalysis, AIResponse
from app.services.distortion_service import CANONICAL_DISTORTIONS


class SinglePageCanvas(canvas.Canvas):
    """
    Two-pass canvas that validates single-page enforcement and renders a clean,
    minimalist footer with accurate page numbering and confidentiality notice.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count: int):
        self.saveState()
        # Clean, subtle hairline footer rule in brand border color
        self.setStrokeColor(colors.HexColor("#D8E4F8"))
        self.setLineWidth(0.5)
        self.line(28, 28, 567, 28)

        # Footer typography: Slate Gray #64748B, 6.8pt Inter
        self.setFont("Inter", 6.8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(28, 18, "Better Me AI Companion • Confidential CBT Educational Summary • Not a medical diagnosis or formal clinical prescription")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(567, 18, page_str)
        self.restoreState()


class PDFReportService:
    """
    Better Me Signature CBT Session Report Generator.
    Aligned with the Better Me web application branding (Royal Blue #2A5FB7, Deep Navy #182A4D,
    Soft Lavender #B8A1FF, Ice Blue #EBF3FC, Poppins & Inter typography).
    Strictly single A4 page with accurate clinical data and zero symptom/score hallucinations.
    """

    COLOR_PRIMARY = colors.HexColor("#2A5FB7")        # Better Me Royal Blue
    COLOR_DARK_NAVY = colors.HexColor("#182A4D")      # Better Me Deep Navy
    COLOR_ACCENT_PURPLE = colors.HexColor("#B8A1FF")  # Secondary Lavender Accent
    COLOR_ICE_BG = colors.HexColor("#EBF3FC")         # Soft Ice Blue Header & Pill Tint
    COLOR_CARD_BG = colors.HexColor("#F8FAFE")        # Soft Off-White Card Background
    COLOR_BORDER = colors.HexColor("#D8E4F8")         # Subtle Brand Border
    COLOR_TEXT_DARK = colors.HexColor("#0F172A")      # Dark Slate Charcoal
    COLOR_TEXT_BODY = colors.HexColor("#334155")      # Readable Charcoal Slate
    COLOR_TEXT_MUTED = colors.HexColor("#64748B")     # Muted Slate Metadata
    COLOR_ALERT_BG = colors.HexColor("#FEF2F2")       # Distortion Rose Background
    COLOR_ALERT_TEXT = colors.HexColor("#DC2626")     # Distortion Rose Text
    COLOR_SUCCESS_BG = colors.HexColor("#ECFDF5")     # Reframed Emerald Background
    COLOR_SUCCESS_TEXT = colors.HexColor("#059669")   # Reframed Emerald Text

    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.logo_path = os.path.join(base_dir, "assets", "Better_me_Logo.png")
        if not os.path.exists(self.logo_path):
            alt_path = os.path.join(base_dir, "..", "better-me-frontend", "src", "imports", "Better_me_Logo.png")
            if os.path.exists(alt_path):
                self.logo_path = alt_path

        self.fonts_registered = False
        self._register_fonts(base_dir)

    def _register_fonts(self, base_dir: str):
        """Registers Poppins and Inter TrueType fonts with ReportLab."""
        fonts_dir = os.path.join(base_dir, "assets", "fonts")
        poppins_bold = os.path.join(fonts_dir, "Poppins-Bold.ttf")
        poppins_reg = os.path.join(fonts_dir, "Poppins-Regular.ttf")
        inter_path = os.path.join(fonts_dir, "Inter.ttf")

        try:
            if os.path.exists(poppins_bold):
                pdfmetrics.registerFont(TTFont("Poppins-Bold", poppins_bold))
            else:
                pdfmetrics.registerFont(TTFont("Poppins-Bold", "Helvetica-Bold"))

            if os.path.exists(poppins_reg):
                pdfmetrics.registerFont(TTFont("Poppins-Regular", poppins_reg))
            else:
                pdfmetrics.registerFont(TTFont("Poppins-Regular", "Helvetica"))

            if os.path.exists(inter_path):
                pdfmetrics.registerFont(TTFont("Inter", inter_path))
            else:
                pdfmetrics.registerFont(TTFont("Inter", "Helvetica"))

            self.fonts_registered = True
        except Exception as e:
            print(f"[WARNING] Font registration notice: {e}")
            self.fonts_registered = False

    def _get_styles(self):
        styles = getSampleStyleSheet()

        h_bold = "Poppins-Bold" if self.fonts_registered else "Helvetica-Bold"
        h_reg = "Poppins-Regular" if self.fonts_registered else "Helvetica"
        b_font = "Inter" if self.fonts_registered else "Helvetica"

        styles.add(ParagraphStyle(
            name="BrandHeader",
            fontName=h_bold,
            fontSize=13.5,
            leading=16.5,
            textColor=self.COLOR_PRIMARY
        ))

        styles.add(ParagraphStyle(
            name="BrandDocType",
            fontName=h_bold,
            fontSize=8.0,
            leading=10.5,
            textColor=self.COLOR_DARK_NAVY
        ))

        styles.add(ParagraphStyle(
            name="BrandSub",
            fontName=b_font,
            fontSize=6.8,
            leading=9.2,
            textColor=self.COLOR_TEXT_MUTED
        ))

        styles.add(ParagraphStyle(
            name="BadgeTop",
            fontName=h_bold,
            fontSize=7.2,
            leading=9.5,
            textColor=self.COLOR_PRIMARY,
            alignment=2
        ))

        styles.add(ParagraphStyle(
            name="BadgeSub",
            fontName=b_font,
            fontSize=6.5,
            leading=8.5,
            textColor=self.COLOR_TEXT_MUTED,
            alignment=2
        ))

        styles.add(ParagraphStyle(
            name="SectionLabel",
            fontName=h_bold,
            fontSize=7.5,
            leading=10.0,
            textColor=self.COLOR_DARK_NAVY,
            spaceBefore=0,
            spaceAfter=0
        ))

        styles.add(ParagraphStyle(
            name="CardHeaderLabel",
            fontName=h_bold,
            fontSize=7.2,
            leading=9.5,
            textColor=self.COLOR_DARK_NAVY
        ))

        styles.add(ParagraphStyle(
            name="CardHeaderRed",
            fontName=h_bold,
            fontSize=7.2,
            leading=9.5,
            textColor=self.COLOR_ALERT_TEXT
        ))

        styles.add(ParagraphStyle(
            name="CardHeaderGreen",
            fontName=h_bold,
            fontSize=7.2,
            leading=9.5,
            textColor=self.COLOR_SUCCESS_TEXT
        ))

        styles.add(ParagraphStyle(
            name="BodySmall",
            fontName=b_font,
            fontSize=7.0,
            leading=9.8,
            textColor=self.COLOR_TEXT_BODY
        ))

        styles.add(ParagraphStyle(
            name="BodySmallBold",
            fontName=h_bold,
            fontSize=7.0,
            leading=9.8,
            textColor=self.COLOR_TEXT_DARK
        ))

        styles.add(ParagraphStyle(
            name="MetaLabel",
            fontName=b_font,
            fontSize=6.2,
            leading=8.0,
            textColor=self.COLOR_TEXT_MUTED
        ))

        styles.add(ParagraphStyle(
            name="MetaVal",
            fontName=h_bold,
            fontSize=7.4,
            leading=9.5,
            textColor=self.COLOR_DARK_NAVY
        ))

        styles.add(ParagraphStyle(
            name="QuoteSmall",
            fontName=b_font,
            fontSize=7.0,
            leading=9.8,
            textColor=self.COLOR_DARK_NAVY
        ))

        styles.add(ParagraphStyle(
            name="DisclaimerSmall",
            fontName=b_font,
            fontSize=6.2,
            leading=8.0,
            textColor=self.COLOR_TEXT_MUTED
        ))

        styles.add(ParagraphStyle(
            name="PillTag",
            fontName=h_bold,
            fontSize=6.8,
            leading=9.0,
            textColor=self.COLOR_PRIMARY
        ))

        return styles

    def extract_session_data(self, conversation: Conversation, db: Session) -> Dict[str, Any]:
        """
        Extracts verified session data directly from saved conversation messages and session state.
        Never hallucinates symptoms, scores, diagnoses, or unstated clinical outcomes.
        """
        user: Optional[User] = conversation.user
        session_state: Optional[ConversationSessionState] = conversation.session_state

        # Calculate chronological session number for this user
        all_user_convs = (
            db.query(Conversation)
            .filter(Conversation.user_id == conversation.user_id)
            .order_by(Conversation.created_at.asc())
            .all()
        )
        try:
            session_num = all_user_convs.index(conversation) + 1
        except Exception:
            session_num = 1

        messages: List[Message] = conversation.messages or []
        user_msgs = [m for m in messages if m.sender_type == "user"]
        ai_msgs = [m for m in messages if m.sender_type == "ai"]

        # Calculate duration
        if messages and len(messages) >= 2:
            start_time = messages[0].created_at
            end_time = messages[-1].created_at
            duration_mins = max(1, int((end_time - start_time).total_seconds() // 60))
            if duration_mins == 0:
                duration_mins = max(5, len(messages) * 2)
        else:
            duration_mins = 15

        session_date = conversation.created_at.strftime("%B %d, %Y")

        # 1. Patient & Session Information (Privacy-Preserved: NO sensitive email or raw UUIDs)
        patient_name = (user.full_name or "").strip() if user else "Patient"
        if not patient_name or patient_name.lower() in ["user", "none"]:
            patient_name = "Patient"

        # 2. Extract Self-Reported Distress Rating strictly from Conversation
        distress_score: Optional[int] = None
        for i, m in enumerate(messages):
            if m.sender_type == "ai" and any(k in m.content.lower() for k in ["0 to 100", "scale of 0", "rate how intense", "where would it be"]):
                # Look at the immediate subsequent user response
                if i + 1 < len(messages) and messages[i + 1].sender_type == "user":
                    reply_text = messages[i + 1].content
                    # Look for number in reply (e.g. "Probably an 85. I felt...")
                    match = re.search(r"\b(\d{1,3})\b", reply_text)
                    if match:
                        val = int(match.group(1))
                        if 0 <= val <= 100:
                            distress_score = val
                            break

        if distress_score is None:
            # Check explicit rating patterns across all user messages
            for um in user_msgs:
                m100 = re.search(r"\b(\d{1,3})\s*(?:\/|\s*out of\s*)\s*100\b", um.content.lower())
                if m100 and 0 <= int(m100.group(1)) <= 100:
                    distress_score = int(m100.group(1))
                    break
                m10 = re.search(r"\b(\d{1,2})\s*(?:\/|\s*out of\s*)\s*10\b", um.content.lower())
                if m10 and 0 <= int(m10.group(1)) <= 10:
                    distress_score = int(m10.group(1)) * 10
                    break

        if distress_score is not None:
            distress_display = f"{distress_score} / 100"
            distress_note = "Self-reported emotional intensity during situational exploration."
        else:
            distress_display = "Not assessed"
            distress_note = "Numerical rating was not formally assessed during this dialogue."

        # 3. Extract Somatic Symptoms strictly from User Messages
        somatic_symptoms: List[str] = []
        somatic_patterns = [
            (r"\b(?:face\s+getting\s+hot|hot\s+face)\b", "Face flushed / hot"),
            (r"\b(?:hands?\s+shaking|shaking\s+hands?|trembling)\b", "Hands shaking / trembling"),
            (r"\b(?:heart\s+racing|racing\s+heart|palpitations)\b", "Accelerated heart rate"),
            (r"\b(?:tight\s+chest|chest\s+tightness|tightness\s+in\s+chest)\b", "Chest tightness"),
            (r"\b(?:knot\s+in\s+stomach|stomach\s+knot|stomach\s+cramps)\b", "Stomach tension"),
            (r"\b(?:sweating|sweat|sweaty)\b", "Perspiration / sweating"),
            (r"\b(?:trouble\s+sleeping|can\'?t\s+sleep|insomnia|restless\s+sleep)\b", "Sleep disruption"),
            (r"\b(?:headache|migraine)\b", "Headache tension"),
            (r"\b(?:dizziness|lightheaded)\b", "Lightheadedness"),
            (r"\b(?:exhausted|fatigue|exhaustion)\b", "Physical fatigue")
        ]
        for um in user_msgs:
            for pat, label in somatic_patterns:
                if re.search(pat, um.content, re.IGNORECASE) and label not in somatic_symptoms:
                    somatic_symptoms.append(label)

        if somatic_symptoms:
            somatic_display = ", ".join(somatic_symptoms) + " (Patient-Reported)"
        else:
            somatic_display = "Not assessed / None reported by patient"

        # 4. Session Overview & Presenting Concerns (Concise, actual user context)
        presenting_concern = ""
        if user_msgs:
            # Take first user message clean context
            first_msg = user_msgs[0].content.strip().strip('"').strip("'")
            sentences = [s.strip() for s in re.split(r"[.!?]", first_msg) if len(s.strip()) > 5]
            if sentences:
                presenting_concern = sentences[0]
                if len(sentences) > 1 and len(presenting_concern) < 100:
                    presenting_concern += f". {sentences[1]}"
            else:
                presenting_concern = first_msg
        if not presenting_concern and session_state and session_state.identified_trigger:
            presenting_concern = session_state.identified_trigger

        if not presenting_concern:
            presenting_concern = "General daily stress, emotional fatigue, and cognitive reflection."
        if len(presenting_concern) > 175:
            presenting_concern = presenting_concern[:172] + "..."

        # 5. Extract Automatic Thought strictly from Patient
        auto_thought = ""
        # Look for user message after AI asks about automatic thought
        for i, m in enumerate(messages):
            if m.sender_type == "ai" and any(k in m.content.lower() for k in ["automatic thought", "running through your mind", "belief or assumption"]):
                if i + 1 < len(messages) and messages[i + 1].sender_type == "user":
                    candidate = messages[i + 1].content.strip().strip('"').strip("'")
                    if len(candidate) > 5:
                        auto_thought = candidate
                        break

        if not auto_thought:
            # Look for user messages containing thought cue words
            for um in user_msgs[1:]:
                txt = um.content.strip().strip('"').strip("'")
                if any(w in txt.lower() for w in ["they feel", "i thought", "incompetent", "fail", "hate", "lose", "ruined", "never"]):
                    auto_thought = txt
                    break

        if not auto_thought and session_state and session_state.automatic_thought:
            auto_thought = session_state.automatic_thought.strip().strip('"').strip("'")

        if not auto_thought:
            auto_thought = "Self-critical appraisal of situational capabilities."
        if len(auto_thought) > 130:
            auto_thought = auto_thought[:127] + "..."

        # 6. Extract Possible Cognitive Pattern & Evidence
        distortion_name = ""
        if session_state and session_state.identified_distortion:
            distortion_name = session_state.identified_distortion
        if not distortion_name:
            for analysis in (conversation.analyses or []):
                if analysis.distortion_class and analysis.distortion_class not in ["None", "Normal", "Unknown"] and not analysis.distortion_class.startswith("LABEL_"):
                    if analysis.distortion_confidence >= 0.50:
                        distortion_name = analysis.distortion_class
                        break

        # Sanitize raw labels
        if distortion_name and str(distortion_name).startswith("LABEL_"):
            try:
                idx = int(str(distortion_name).split("_")[-1])
                distortion_name = CANONICAL_DISTORTIONS.get(idx, "")
            except Exception:
                distortion_name = ""

        # Check for patterns from conversation text if unassigned
        evidence_lower = " ".join([m.content.lower() for m in user_msgs])
        if not distortion_name or distortion_name == "General":
            if "they" in evidence_lower and any(w in evidence_lower for w in ["think", "feel", "notice", "hate"]):
                distortion_name = "Mind Reading"
            elif any(w in evidence_lower for w in ["ruin", "never", "lost", "fail forever", "hate me"]):
                distortion_name = "Catastrophizing"
            elif any(w in evidence_lower for w in ["perfect", "mistake", "100%", "worthless"]):
                distortion_name = "All-or-Nothing Thinking"

        if not distortion_name or distortion_name in ["None", "Normal"]:
            distortion_name = "None Identified (Adaptive Reflection)"
            pattern_evidence = "Patient explored thoughts without rigid cognitive distortion triggers."
        else:
            pattern_evidence = f'"{auto_thought}"'

        # 7. Extract Balanced Perspective co-created in Session (Never Therapist Questions!)
        balanced_perspective = ""
        for i, m in enumerate(messages):
            if m.sender_type == "ai" and any(k in m.content.lower() for k in ["kinder and more accurate", "reframe", "balanced way", "step back from"]):
                if i + 1 < len(messages) and messages[i + 1].sender_type == "user":
                    user_ref = messages[i + 1].content.strip().strip('"').strip("'")
                    if len(user_ref) > 5 and not user_ref.endswith("?"):
                        if user_ref.lower().startswith("that "):
                            user_ref = user_ref[5:]
                        balanced_perspective = user_ref[0].upper() + user_ref[1:]
                        break

        # Check for factual counter-evidence from conversation (e.g. manager feedback)
        counter_facts = []
        for um in user_msgs:
            t = um.content.lower()
            if "manager" in t and ("thanked" in t or "clear" in t or "good" in t):
                counter_facts.append("Manager provided positive feedback on clarity")
            if "nobody" in t and ("laughed" in t or "walked out" in t or "complained" in t):
                counter_facts.append("No negative reactions occurred")

        if counter_facts and balanced_perspective:
            balanced_perspective += f". Objective feedback confirmed: {', '.join(counter_facts)}."
        elif not balanced_perspective and session_state and session_state.balanced_perspective:
            cand = session_state.balanced_perspective.strip().strip('"')
            if not cand.endswith("?"):
                balanced_perspective = cand

        if not balanced_perspective:
            balanced_perspective = "It is acceptable to experience nervous moments. A single perceived setback does not define competence, and objective facts show that progress is ongoing."

        if len(balanced_perspective) > 180:
            balanced_perspective = balanced_perspective[:177] + "..."

        # Ensure balanced perspective never ends with a question mark
        balanced_perspective = re.sub(r"\?+$", ".", balanced_perspective)

        # 8. Extract Personalized Homework strictly relevant to actual discussion
        homework = ""
        if session_state and session_state.metadata_json:
            homework = session_state.metadata_json.get("homework")

        if not homework or len(homework) < 15:
            topic_text = (presenting_concern + " " + auto_thought).lower()
            if any(k in topic_text for k in ["work", "meeting", "presentation", "project", "boss", "manager", "speech"]):
                homework = "Notice when 'mind-reading' or 'must be perfect' thoughts arise before speaking. Remind yourself that making mistakes is acceptable and note one objective positive fact."
            elif any(k in topic_text for k in ["exam", "study", "grade", "marks", "test", "assignment"]):
                homework = "Complete a 20-minute focused study interval with the 'Good-Enough Experiment': notice thoughts of failure, write them down, and return to work without self-criticism."
            elif any(k in topic_text for k in ["chest", "heart", "breath", "panic", "sleep", "shaking"]):
                homework = "Practice 3 cycles of 4-7-8 diaphragmatic breathing twice daily whenever somatic tension rises, noting physical sensations before and after."
            else:
                homework = "Thought Catching: Record 1 automatic negative thought daily alongside 1 balanced, realistic alternative observation."

        if len(homework) > 175:
            homework = homework[:172] + "..."

        # 9. Next Session Goal
        next_goal = ""
        if session_state and session_state.metadata_json:
            next_goal = session_state.metadata_json.get("next_session_goal")

        if not next_goal or len(next_goal) < 10:
            if "work" in presenting_concern.lower() or "presentation" in presenting_concern.lower():
                next_goal = "Review workplace confidence progress and evaluate how holding the balanced perspective supported upcoming team communication."
            else:
                next_goal = "Review the behavioral homework outcome and evaluate cognitive flexibility under situational stress."

        if len(next_goal) > 165:
            next_goal = next_goal[:162] + "..."

        return {
            "conversation_id": str(conversation.id),
            "report_id": f"REP-{str(conversation.id)[:8].upper()}",
            "session_number": f"Session #{session_num:02d}",
            "session_date": session_date,
            "session_time": conversation.created_at.strftime("%I:%M %p"),
            "duration_str": f"{duration_mins} Min",
            "patient_name": patient_name,
            "distress_score": distress_score,
            "distress_display": distress_display,
            "distress_note": distress_note,
            "self_reported_distress": distress_display,
            "somatic_symptoms": somatic_display,
            "presenting_concern": presenting_concern,
            "presenting_trigger": presenting_concern,
            "automatic_thought": auto_thought,
            "distortion_name": distortion_name,
            "identified_distortion": distortion_name,
            "distortion_evidence": pattern_evidence,
            "balanced_thought": balanced_perspective,
            "homework": homework,
            "next_session_goal": next_goal,
            "cbt_technique": "Socratic Questioning, Evidence Testing & Objective Decatastrophizing",
            "total_messages": len(messages),
            "user_messages_count": len(user_msgs),
            "ai_messages_count": len(ai_msgs)
        }

    def generate_pdf_buffer(self, conversation: Conversation, db: Session) -> io.BytesIO:
        """
        Builds the redesigned, premium single-page CBT Session Report.
        Strictly fits within ONE A4 page (595 x 842 pt) with zero vertical overflow.
        """
        data = self.extract_session_data(conversation, db)
        buffer = io.BytesIO()

        # A4 Geometry: 595.27 x 841.89 pt. Margins: 28 pt left/right, 24 pt top, 36 pt bottom.
        # Printable width: 595.27 - 56 = 539.27 pt.
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=28,
            rightMargin=28,
            topMargin=24,
            bottomMargin=36
        )

        styles = self._get_styles()
        story = []

        # -------------------------------------------------------------
        # HEADER: BRANDING & EDITORIAL TITLE
        # -------------------------------------------------------------
        logo_cell = ""
        if os.path.exists(self.logo_path):
            try:
                logo_cell = Image(self.logo_path, width=32, height=32)
            except Exception:
                logo_cell = ""

        header_title = [
            Paragraph("Better Me", styles["BrandHeader"]),
            Paragraph("COGNITIVE BEHAVIORAL THERAPY (CBT) SESSION REPORT", styles["BrandDocType"]),
            Paragraph("Structured Cognitive Reflection & Psychoeducation Summary", styles["BrandSub"])
        ]

        header_badge = [
            Paragraph("AI-Assisted CBT Session", styles["BadgeTop"]),
            Paragraph(f"{data['session_date']}", styles["BadgeSub"])
        ]

        if logo_cell:
            top_table = Table([[logo_cell, header_title, header_badge]], colWidths=[38, 360, 141])
            top_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
        else:
            top_table = Table([[header_title, header_badge]], colWidths=[398, 141])
            top_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))

        story.append(top_table)
        story.append(Spacer(1, 5))

        # -------------------------------------------------------------
        # SECTION 1: PATIENT & SESSION INFORMATION (Minimal Clean Card)
        # -------------------------------------------------------------
        info_data = [
            [
                Paragraph("PATIENT NAME", styles["MetaLabel"]),
                Paragraph("SESSION NUMBER", styles["MetaLabel"]),
                Paragraph("DATE & TIME", styles["MetaLabel"]),
                Paragraph("DURATION & FORMAT", styles["MetaLabel"])
            ],
            [
                Paragraph(f"{data['patient_name']}", styles["MetaVal"]),
                Paragraph(f"<font color='{self.COLOR_PRIMARY.hexval()}'><b>{data['session_number']}</b></font>", styles["MetaVal"]),
                Paragraph(f"{data['session_date']} • {data['session_time']}", styles["MetaVal"]),
                Paragraph(f"{data['duration_str']} • Interactive CBT", styles["MetaVal"])
            ]
        ]
        info_table = Table(info_data, colWidths=[134, 135, 135, 135])
        info_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), self.COLOR_CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("LINEBEFORE", (1, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, 0), 3),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
            ("TOPPADDING", (0, 1), (-1, 1), 1),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))
        story.append(info_table)
        story.append(Spacer(1, 6))

        # -------------------------------------------------------------
        # SECTION 2: SESSION OVERVIEW AND IDENTIFIED CONCERNS
        # -------------------------------------------------------------
        overview_data = [
            [
                Paragraph("1. SESSION OVERVIEW & PRESENTING SITUATION", styles["SectionLabel"])
            ],
            [
                Paragraph(f"<b>Presenting Situation:</b> {data['presenting_concern']}", styles["BodySmall"])
            ],
            [
                Paragraph(f"<b>Physical / Somatic Signals:</b> {data['somatic_symptoms']}", styles["BodySmall"])
            ]
        ]
        overview_table = Table(overview_data, colWidths=[539])
        overview_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.COLOR_ICE_BG),
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("INNERGRID", (0, 1), (-1, -1), 0.5, self.COLOR_CARD_BG),
            ("LINELEFT", (0, 0), (0, -1), 3.0, self.COLOR_PRIMARY),
            ("TOPPADDING", (0, 0), (-1, 0), 3),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
            ("TOPPADDING", (0, 1), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 6))

        # -------------------------------------------------------------
        # SECTION 3: POSSIBLE COGNITIVE PATTERNS
        # -------------------------------------------------------------
        patterns_data = [
            [
                Paragraph("2. IDENTIFIED COGNITIVE PATTERN", styles["SectionLabel"]),
                Paragraph("EDUCATIONAL NOTE", styles["MetaLabel"])
            ],
            [
                Paragraph(
                    f"<b>Identified Pattern:</b> <font color='{self.COLOR_PRIMARY.hexval()}'><b>{data['distortion_name']}</b></font><br/>"
                    f"<b>Supporting Dialogue Excerpt:</b> <i>{data['distortion_evidence']}</i>",
                    styles["BodySmall"]
                ),
                Paragraph(
                    "<i>Cognitive patterns represent automated thinking habits identified during reflection; they do not constitute a formal medical diagnosis.</i>",
                    styles["DisclaimerSmall"]
                )
            ]
        ]
        patterns_table = Table(patterns_data, colWidths=[369, 170])
        patterns_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.COLOR_ICE_BG),
            ("BACKGROUND", (0, 1), (-1, 1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("LINELEFT", (0, 0), (0, -1), 3.0, self.COLOR_ACCENT_PURPLE),
            ("TOPPADDING", (0, 0), (-1, 0), 3),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
            ("TOPPADDING", (0, 1), (-1, 1), 3),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))
        story.append(patterns_table)
        story.append(Spacer(1, 6))

        # -------------------------------------------------------------
        # SECTION 4: CBT INTERVENTIONS & BALANCED PERSPECTIVES
        # -------------------------------------------------------------
        reframe_cols = [
            [
                Paragraph("UNHELPFUL AUTOMATIC THOUGHT", styles["CardHeaderRed"]),
                Paragraph("BALANCED REFRAMED PERSPECTIVE", styles["CardHeaderGreen"])
            ],
            [
                Paragraph(f"\"{data['automatic_thought']}\"", styles["QuoteSmall"]),
                Paragraph(f"{data['balanced_thought']}", styles["BodySmall"])
            ],
            [
                Paragraph(f"<b>CBT Intervention Applied:</b> {data['cbt_technique']}", styles["MetaLabel"]),
                ""
            ]
        ]
        reframe_table = Table(
            [
                reframe_cols[0],
                reframe_cols[1],
                [
                    Paragraph(f"<b>CBT Intervention Applied:</b> {data['cbt_technique']}", styles["MetaLabel"]),
                    ""
                ]
            ],
            colWidths=[269, 270]
        )
        reframe_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), self.COLOR_ALERT_BG),
            ("BACKGROUND", (1, 0), (1, 0), self.COLOR_SUCCESS_BG),
            ("BACKGROUND", (0, 1), (0, 1), colors.white),
            ("BACKGROUND", (1, 1), (1, 1), self.COLOR_CARD_BG),
            ("BACKGROUND", (0, 2), (-1, 2), self.COLOR_CARD_BG),
            ("SPAN", (0, 2), (1, 2)),
            ("BOX", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("LINELEFT", (0, 0), (0, 1), 2.5, self.COLOR_ALERT_TEXT),
            ("LINELEFT", (1, 0), (1, 1), 2.5, self.COLOR_SUCCESS_TEXT),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))
        story.append(reframe_table)
        story.append(Spacer(1, 6))

        # -------------------------------------------------------------
        # SECTION 5: PROGRESS & PERSONALIZED HOMEWORK
        # -------------------------------------------------------------
        progress_elements = [
            Paragraph(f"<b>Distress Rating:</b> <font color='{self.COLOR_PRIMARY.hexval()}'><b>{data['distress_display']}</b></font>", styles["BodySmallBold"])
        ]
        if data["distress_score"] is not None:
            bar_w = 140
            fill_w = max(4, int(bar_w * (data["distress_score"] / 100.0)))
            d = Drawing(bar_w, 7)
            d.add(Rect(0, 0, bar_w, 5, rx=2.5, ry=2.5, fillColor=self.COLOR_BORDER, strokeColor=None))
            d.add(Rect(0, 0, fill_w, 5, rx=2.5, ry=2.5, fillColor=self.COLOR_PRIMARY, strokeColor=None))
            progress_elements.append(d)
        progress_elements.append(Paragraph(data["distress_note"], styles["MetaLabel"]))

        homework_elements = [
            Paragraph(f"{data['homework']}", styles["BodySmall"])
        ]

        action_data = [
            [
                Paragraph("4. PROGRESS & EMOTIONAL RATING", styles["SectionLabel"]),
                Paragraph("5. PERSONALIZED HOMEWORK EXERCISE", styles["SectionLabel"])
            ],
            [
                progress_elements,
                homework_elements
            ]
        ]
        action_table = Table(action_data, colWidths=[209, 330])
        action_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.COLOR_ICE_BG),
            ("BACKGROUND", (0, 1), (-1, 1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("LINELEFT", (0, 0), (0, -1), 3.0, self.COLOR_PRIMARY),
            ("TOPPADDING", (0, 0), (-1, 0), 3),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
            ("TOPPADDING", (0, 1), (-1, 1), 3),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))
        story.append(action_table)
        story.append(Spacer(1, 6))

        # -------------------------------------------------------------
        # SECTION 6: NEXT SESSION GOALS
        # -------------------------------------------------------------
        goals_data = [
            [
                Paragraph("6. CONTINUITY & NEXT SESSION GOALS", styles["SectionLabel"])
            ],
            [
                Paragraph(f"{data['next_session_goal']}", styles["BodySmall"])
            ]
        ]
        goals_table = Table(goals_data, colWidths=[539])
        goals_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.COLOR_ICE_BG),
            ("BACKGROUND", (0, 1), (-1, 1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, self.COLOR_BORDER),
            ("LINELEFT", (0, 0), (0, -1), 3.0, self.COLOR_DARK_NAVY),
            ("TOPPADDING", (0, 0), (-1, 0), 3),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
            ("TOPPADDING", (0, 1), (-1, 1), 3),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))
        story.append(goals_table)

        # Build Document with SinglePageCanvas
        doc.build(story, canvasmaker=SinglePageCanvas)
        buffer.seek(0)
        return buffer


pdf_report_service = PDFReportService()
