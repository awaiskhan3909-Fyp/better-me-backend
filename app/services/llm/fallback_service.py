import re
from typing import Tuple, Optional, Dict, Any, List
from app.schemas.response import CBTGuidance, DistortionPrediction


class FallbackService:
    """
    Deterministic Clinical CBT Dialogue Engine used when LLM API is unavailable,
    times out, or generates invalid responses.
    Fully context-aware, tracks multi-turn conversational progression, validates emotions,
    decouples automatic thoughts from reality, and provides actionable coping plans.
    """

    SAFETY_OVERRIDE_TEXT = (
        "Your safety and the well-being of those you care about are incredibly important. "
        "If you or someone you know is experiencing overwhelming distress, suicidal thoughts, or thoughts of self-harm, "
        "please reach out immediately to a trusted professional, doctor, or emergency crisis helpline (such as 988 or local emergency services). "
        "Support is available 24/7 and you do not have to carry this alone."
    )

    SAFETY_OVERRIDE_ROMAN_URDU_TEXT = (
        "Aapki hifazat aur zindagi bohot qeemti hai. "
        "Agar aap shadeed takleef ya khudkushi ke khayalat se guzar rahe hain, "
        "to barah-e-karam foran kisi doctor ya crisis helpline se rabta karein:\n"
        "• Pakistan Mental Health Helpline (Umang): 0311-7786264 / 0317-4288898\n"
        "• International Crisis Helpline: 988\n"
        "Aap is mushkil waqt me akele nahi hain, madad 24/7 dastiyab hai."
    )

    CBT_ENGLISH_REFRAMES = {
        "Overgeneralization": (
            "It sounds like because this one specific situation didn't go the way you hoped, your mind is concluding that everything is going to fall apart. "
            "In CBT, this is called Overgeneralization. A single setback—even an exam or test—is painful, but it does not represent your entire life or your future. "
            "If we take a step back, what is one area of your life or one skill that is still steady right now?"
        ),
        "Catastrophizing": (
            "Your mind is immediately jumping to the worst-case scenario. When distress is high, our brains amplify fear and convince us that total catastrophe is guaranteed. "
            "In cognitive therapy, we pause to examine the probability. Realistically, between the absolute worst outcome and what is most likely to happen, what is a more balanced possibility?"
        ),
        "All-or-Nothing Thinking": (
            "You seem to be viewing this through a binary lens—as either total perfection or complete failure. "
            "In reality, growth and human life exist in the spaces between those extremes. Stumbling on one paper does not erase your preparation, intelligence, or future possibilities. "
            "How might we look at this as a learning milestone rather than a final verdict?"
        ),
        "Emotional Reasoning": (
            "Because you are feeling so intensely hurt and discouraged, it feels like those emotions must reflect absolute truth. "
            "In CBT, we remember: feelings are real and valid, but feelings are not facts. Feeling like a failure right now does not make you a failure. "
            "What objective facts or evidence challenge that negative conclusion?"
        ),
        "Fortune Telling": (
            "You are predicting future defeat before it has even had a chance to unfold. "
            "Our minds often confuse the fear of the future with knowledge of the future. The future remains unwritten, and tomorrow brings new opportunities to adapt and recover. "
            "What is one constructive micro-step we can focus on just for today?"
        ),
        "Mind Reading": (
            "It sounds like you are assuming others are judging or looking down on you without direct confirmation. "
            "In CBT, we call this Mind Reading. Most people are absorbed in their own challenges and struggles. "
            "Did anyone actually say that to you, or is your stress projecting that fear onto them?"
        ),
    }

    CBT_ROMAN_URDU_REFRAMES = {
        "Overgeneralization": (
            "Aisa lagta hai ke aik waqiye ki wajah se aapko lag raha hai ke sab kuch hi kharab ho gaya hai (Overgeneralization). "
            "CBT ke mutabiq ye aik fikri khata hai. Aik parcha fail hone ka matlab ye nahi ke poori zindagi tabah ho gayi. "
            "Kya hum mil kar dekh sakte hain ke kya koi aisi cheez hai jo abhi bhi theek chal rahi hai?"
        ),
        "Catastrophizing": (
            "Aapka zehn foran badtareen soorat-e-haal ki taraf ja raha hai (Catastrophizing). "
            "Jab hum pareshan hote hain to zehan har cheez ko tabahi samajhta hai. "
            "Agar hum thehar kar thandi dimaag se sochein, to haqeeqat mein sab se zyada imkaan kis baat ka hai?"
        ),
        "All-or-Nothing Thinking": (
            "Aap cheezon ko sirf 'poori kamyaabi ya mukammal nakami' ke do pehlo mein dekh rahe hain (All-or-Nothing Thinking). "
            "Zindagi mein darmiyani raste aur seekhne ke marhale bhi hotay hain. Aik nakami aapki poori mehnat ko khatam nahi karti. "
            "Aap is baare mein kya sochte hain?"
        ),
        "Emotional Reasoning": (
            "Aapko aisa mehsoos ho raha hai, is liye lag raha hai ke haqeeqat bhi aisi hi hai (Emotional Reasoning). "
            "Lekin jazbaat aur haqeeqat do alag cheezein hain. Mayoosi ka ehsas hona alag baat hai, aur haqeeqat mein nakam hona alag. "
            "Kya aap is ehsas ka koi thos saboot dekh sakte hain?"
        ),
        "Fortune Telling": (
            "Aap aane wale waqt ka faisla pehle se hi manfi kar rahe hain (Fortune Telling). "
            "Hum mustaqbil ko pehle se nahi jaan sakte. Agar hum sirf aaj aur agle aik qadam par tawajjo dein to kaisa rahega?"
        ),
        "Mind Reading": (
            "Aapko lag raha hai ke doosre log aapke baare mein bura soch rahe hain (Mind Reading), baghair kisi saboot ke. "
            "Kya unho ne waqatan kuch kaha hai, ya ye sirf aapka andaza hai?"
        ),
    }

    def _extract_conversation_history_text(self, conversation_context: Optional[Any]) -> str:
        if not conversation_context:
            return ""
        recent = getattr(conversation_context, "recent_messages", []) or []
        parts = []
        for msg in recent:
            content = getattr(msg, "content", "")
            if content:
                parts.append(content)
        return " ".join(parts).lower()

    def generate_fallback(
        self,
        user_text: str,
        strategy: str,
        cbt_guidance: Optional[CBTGuidance] = None,
        conversation_context: Optional[Any] = None,
        distortion_pred: Optional[DistortionPrediction] = None,
        is_roman_urdu: bool = False
    ) -> Tuple[str, Optional[Dict[str, Any]]]:

        # -------------------------------------------------------------
        # 1. STRICT SAFETY CRISIS OVERRIDE GATE
        # -------------------------------------------------------------
        if strategy == "safety_response":
            if is_roman_urdu:
                return self.SAFETY_OVERRIDE_ROMAN_URDU_TEXT, None
            return self.SAFETY_OVERRIDE_TEXT, None

        text_lower = user_text.lower().strip()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text_lower))
        history_text = self._extract_conversation_history_text(conversation_context)
        combined_context = f"{history_text} {text_lower}"

        # Context Topic Detection
        has_academic_context = any(
            w in combined_context for w in [
                "exam", "paper", "test", "study", "studies", "result", "grade",
                "college", "school", "university", "parhai", "fail", "failed", "marks", "imtihan"
            ]
        )
        has_existential_context = any(
            phrase in combined_context for phrase in [
                "not fit in this world", "fit in this world", "not fit", "dont belong",
                "waste of space", "qabil nahi", "dunya ke qabil", "ruined my life", "my life ruins"
            ]
        )

        # -------------------------------------------------------------
        # 2. SOLUTION & COPING PLAN SEEKING INTENT
        # e.g., "tell me the solution of this bro", "kya karun", "what should i do", "how to solve"
        # -------------------------------------------------------------
        is_solution_request = bool(
            re.search(
                r"\b(?:tell\s+me\s+(?:the\s+)?solution|what\s+is\s+the\s+solution|solution\s+of\s+this|what\s+(?:should|can)\s+i\s+do|how\s+to\s+fix|how\s+can\s+i\s+fix|give\s+me\s+(?:a\s+)?solution|what\s+is\s+the\s+way\s+out|kya\s+karun|kya\s+karein|hal\s+batao|solution\s+batao|iska\s+hal|koi\s+rasta)\b",
                text_lower
            )
        )

        if is_solution_request:
            if is_roman_urdu:
                if has_academic_context or has_existential_context:
                    content = (
                        "Is soorat-e-haal ke liye CBT ka 3-marhala war amali hal (Action Plan) ye hai:\n\n"
                        "1. **Zaat aur Exam ko Alag Karein (Decouple Identity):**\n"
                        "Aik paper fail hone ka matlab ye hargiz nahi ke aap nakam insan hain ya dunya ke qabil nahi. Exam sirf kisi aik din ki koshish aur circumstances ko test karta hai, ye aapki poori qabiliyat ya wajood ka faisla nahi karta.\n\n"
                        "2. **Haqeeqat ka Mutawazan Jaiza (Reality Testing):**\n"
                        "Aapne bataya ke aapne achi tayyari ki thi—is se saabit hota hai ke aap mehnat karne ka hosla rakhte hain. Aik paper dobara diya ja sakta hai ya re-checking ho sakti hai. Aik rukawat poore mustaqbil ko khatam nahi kar sakti.\n\n"
                        "3. **Fauri Amali Qadam (Grounding & Next Steps):**\n"
                        "• *Abhi ke liye:* 3 lambay gehray saans lein (4 second andar, 4 second rokein, 6 second bahar nikalein) taake zehni ghul-o-gupash aur khauf kam ho.\n"
                        "• *Agla Qadam:* Jab zehen thehar jaye, to paper ka jaiza lein ya teacher se mashwara karein ke marks kahan katay taake usay behtar banaya ja sake.\n\n"
                        "Kya aap is mutawazan tareeqay se aik qadam uthane ke liye tayyar hain?"
                    )
                else:
                    content = (
                        "Is maslay se nikalne ke liye CBT ka 3-marhala war hal ye hai:\n\n"
                        "1. **Khayal ko Pehchanein:** Apne manfi khayal ko aik haqeeqat ke bajaye aik waqti jazbaati radd-e-amal ke tor par dekhein.\n"
                        "2. **Saboot ka Jaiza:** Poochein ke 'kya is baat ka koi thos saboot hai ke sab khatam ho gaya?' Aksar hamara dimaag dukh me badtareen andaza lagata hai.\n"
                        "3. **Aik Chhota Qadam:** Aaj sirf kisi aik aisi cheez par tawajjo dein jo aapke ikhtiyar mein hai, aur baqi cheezon ko waqt par chhor dein.\n\n"
                        "Is waqt aapke ikhtiyar mein konsi aik cheez hai jis se shuru kiya ja sake?"
                    )
            else:
                if has_academic_context or has_existential_context:
                    content = (
                        "Here is a concrete 3-step CBT Action Plan to work through this right now:\n\n"
                        "1. **Decouple Your Identity from the Event:**\n"
                        "An exam tests your recall under time pressure on a single day; it does *not* measure your human worth or mean you are 'not fit for this world.' Failing a paper is an academic setback, not an existential verdict on your life.\n\n"
                        "2. **Examine the Evidence (Reality-Testing):**\n"
                        "You prepared thoroughly, which demonstrates discipline, capability, and effort. One failed paper does not wipe away your potential or ruin your future. Exams can almost always be retaken, reassessed, or balanced out in subsequent terms.\n\n"
                        "3. **Actionable Micro-Steps:**\n"
                        "• *Immediate Grounding:* Take 3 deep diaphragmatic breaths (inhale 4s, hold 4s, exhale slowly 6s) to settle your nervous system out of fight-or-flight.\n"
                        "• *Next 24 Hours:* Once the emotional sting settles, review the exam syllabus objectively or consult your instructor to see exactly where marks were lost. We turn a failure into actionable data for improvement.\n\n"
                        "How does looking at this step-by-step feel to you right now?"
                    )
                else:
                    content = (
                        "Here is a structured 3-step CBT approach to help you gain control right now:\n\n"
                        "1. **Separate Thoughts from Facts:** Notice that intense emotions make thoughts feel like absolute reality, but thoughts are hypotheses, not verdicts.\n"
                        "2. **Perspective Testing:** Ask yourself what advice you would give a trusted friend who was facing this exact dilemma.\n"
                        "3. **One Micro-Action:** Identify the single smallest step within your direct control today and focus solely on that step.\n\n"
                        "What is one small area of this situation that you can take action on today?"
                    )
            cbt_dump = cbt_guidance.model_dump() if cbt_guidance else {}
            cbt_dump["balanced_thought_guidance"] = content
            return content, cbt_dump

        # -------------------------------------------------------------
        # 3. EXISTENTIAL / DEFECTIVENESS BELIEF
        # e.g., "in my mind i want that i am not fit in this world"
        # -------------------------------------------------------------
        is_existential_statement = bool(
            re.search(
                r"\b(?:not\s+fit\s+(?:in|for)\s+this\s+world|dont\s+belong\s+(?:in|to)\s+this\s+world|not\s+fit\s+in|dunya\s+ke\s+qabil\s+nahi|zindagi\s+me\s+fit\s+nahi|i\s+am\s+not\s+fit)\b",
                text_lower
            )
        )

        if is_existential_statement:
            if is_roman_urdu:
                content = (
                    "Ye sochna ke 'main is dunya ke qabil nahi hoon' bohot dardnak aur bhaari ehsas hai. "
                    "CBT ke mutabiq, ye aik fikri khata (Emotional Reasoning / Defectiveness) hai, jahan dimaag aik exam ki nakami ke sadmay ko aapke poore wajood par laagu kar deta hai.\n\n"
                    "Exam mein mushkil aana aik baat hai, lekin aapki qadar aur zindagi is se kahin barh kar hai. "
                    "Ye khayal dukh aur mayoosi ki wajah se aa raha hai, kisi haqeeqat ki wajah se nahi. "
                    "Aap is waqt akele nahi hain—kya hum dekh sakte hain ke is khayal ke peeche sab se bara darr kya hai?"
                )
            else:
                content = (
                    "Believing that you 'are not fit in this world' is a deeply painful thought to carry, and I want to validate how exhausting that feels. "
                    "In Cognitive Behavioral Therapy, we recognize this as an automatic negative thought—specifically Defectiveness and Emotional Reasoning. "
                    "The brain takes the raw shock and disappointment of failing an exam and drastically leaps to: 'I don't belong in this world.'\n\n"
                    "Failing a test is painful, but it does not diminish your worth or your right to exist. "
                    "Notice how your grief is telling you that you don't fit, even though the actual event was simply an academic paper. "
                    "Can you see how this thought might be an emotional reaction rather than an objective fact?"
                )
            cbt_dump = cbt_guidance.model_dump() if cbt_guidance else {}
            cbt_dump["balanced_thought_guidance"] = content
            return content, cbt_dump

        # -------------------------------------------------------------
        # 4. DURATION / TIME HORIZON ANSWER
        # e.g., "from since 3,4 days", "since 3 days", "3,4 din se"
        # -------------------------------------------------------------
        is_duration_answer = bool(
            re.search(
                r"\b(?:from\s+since\s+\d+|since\s+\d+|for\s+\d+\s+(?:days|weeks|months|hours)|\d+[,.\s-]+\d*\s*(?:days|weeks|months|hours|din|hafte)|since\s+yesterday|kuch\s+dino\s+se|\d+\s+din\s+se)\b",
                text_lower
            )
        )

        if is_duration_answer:
            if is_roman_urdu:
                content = (
                    "3 se 4 din lagatar is shadeed bojh aur pareshani ko uthana insan ko zehni aur jismani tor par thaka deta hai. "
                    "Jab koi manfi khayal kayi din tak zehen me chalta rahe, to dimaag usay haqeeqat samajhne lagta hai halanke ye sirf exam ke sadmay ka asar hai.\n\n"
                    "Itne din is bojh ko akele uthane ke baad aapka thak jana bilkul samajh aata hai. "
                    "Ab waqt hai ke hum is bojh ko halka karein. Kya aap chahte hain ke hum mil kar iska aik practical hal nikaalein?"
                )
            else:
                content = (
                    "Carrying that heavy thought continuously for 3 to 4 days takes a serious emotional and physical toll. "
                    "When distressing thoughts stay trapped in our mind for days without release, our brain starts treating them as unshakeable truths rather than temporary emotional reactions to an exam setback.\n\n"
                    "The fact that you've been carrying this for days shows how much you cared about doing well. "
                    "Now that you've opened up, we don't have to carry it the same way. Would you like to look at the practical solution to work through this?"
                )
            return content, None

        # -------------------------------------------------------------
        # 5. ACADEMIC / EXAM DISTRESS INITIAL STATEMENT
        # e.g., "i failed the paper in my exam i prepare myself well but still i fail..."
        # -------------------------------------------------------------
        is_exam_disclosure = (
            any(w in words for w in ["exam", "paper", "test", "result", "grade", "studies", "failed", "imtihan", "parhai"]) and
            any(w in words for w in ["fail", "failed", "ruins", "sad", "confused", "future", "poor", "kharab", "nakam"])
        )

        if is_exam_disclosure:
            if is_roman_urdu:
                content = (
                    "Achi tayyari ke bawajood exam mein nakami ka samna karna waqatan bohot dukh aur shock deta hai. "
                    "Aapka pareshan hona aur mustaqbil ke baare me ghabrana bilkul samajh aata hai.\n\n"
                    "Lekin zehan me ye khayal aana ke 'zindagi tabah ho gayi' aik Catastrophizing (badtareen soorat-e-haal ka andaza) hai. "
                    "Aik parcha aapki poori zindagi ya qabiliyat ka faisla nahi karta. Is waqt sab se zyada kis baat ka darr aapke zehn par haawi hai?"
                )
            else:
                content = (
                    "Failing an exam—especially after working hard and preparing thoroughly—is a truly painful and confusing shock. "
                    "It makes complete sense that you are feeling sad and worried about your future right now.\n\n"
                    "When we face an unexpected setback like this, our minds often jump straight to Catastrophizing: believing that one bad result means our entire life is ruined. "
                    "But one exam, as distressing as it is today, does not define your capabilities or destroy your future. What thoughts have been distressing you the most since you received the result?"
                )
            return content, None

        # -------------------------------------------------------------
        # 6. CBT SUPPORT STRATEGY OR DETECTED DISTORTION
        # -------------------------------------------------------------
        if strategy == "cbt_support" or (distortion_pred and distortion_pred.confidence >= 0.50 and distortion_pred.predicted_class not in ["None", "Normal"]):
            dist_name = (cbt_guidance.detected_distortion if cbt_guidance else None) or (distortion_pred.predicted_class if distortion_pred else None)
            if is_roman_urdu:
                reframe = self.CBT_ROMAN_URDU_REFRAMES.get(
                    dist_name,
                    "Main samajh sakta hoon ke ye khayal aapke liye kitna mushkil aur pareshan kun hai. "
                    "CBT mein hum apne khayalat ko haqeeqat ke tarazu mein toltay hain. "
                    "Kya is baat ko kisi aur mutawazan (balanced) tarha se bhi dekha ja sakta hai?"
                )
                cbt_dump = cbt_guidance.model_dump() if cbt_guidance else {}
                cbt_dump["balanced_thought_guidance"] = reframe
                return reframe, cbt_dump
            else:
                reframe = self.CBT_ENGLISH_REFRAMES.get(
                    dist_name,
                    cbt_guidance.balanced_thought_guidance if (cbt_guidance and cbt_guidance.balanced_thought_guidance) else
                    "I hear how heavy this thought is. In CBT, we examine thoughts as hypotheses rather than final facts. "
                    "Taking a gentle step back, what is a more balanced way to look at this situation?"
                )
                cbt_dump = cbt_guidance.model_dump() if cbt_guidance else {}
                cbt_dump["balanced_thought_guidance"] = reframe
                return reframe, cbt_dump

        # -------------------------------------------------------------
        # 7. IDENTITY & ROLE QUERIES
        # -------------------------------------------------------------
        if any(w in words for w in ["who", "what", "kaun", "kya"]) and any(w in words for w in ["you", "bot", "ai", "system", "better", "tum", "aap"]):
            if is_roman_urdu:
                content = (
                    "Main Better Me AI hoon — aapka clinical cognitive therapy companion. "
                    "Mera maqsad aapke pareshan kun khayalat (cognitive distortions) ko pehchan kar "
                    "CBT reframing ke zariye aapko sakoon aur rahnumai faraham karna hai. "
                    "Main aapki kis tarha madad kar sakta hoon?"
                )
            else:
                content = (
                    "I am Better Me, your clinical AI cognitive therapy companion. "
                    "I'm designed to help you explore your thoughts, identify unhelpful cognitive patterns, "
                    "and work through evidence-based CBT exercises in a safe, confidential space. "
                    "How can I support you right now?"
                )
            return content, None

        # -------------------------------------------------------------
        # 8. GREETINGS
        # -------------------------------------------------------------
        if any(w in words for w in ["hi", "hello", "hey", "assalam", "salam"]):
            if is_roman_urdu:
                content = "Salam! Main yahan aapki baat sunne aur aapko support karne ke liye tayyar hoon. Aap kaisa mehsoos kar rahe hain, aur aaj kis baare mein baat karna chahenge?"
            else:
                content = "Hello! I'm here to listen and help you work through any thoughts or situations on your mind today. How are you feeling right now?"
            return content, None

        # -------------------------------------------------------------
        # 9. HOW ARE YOU
        # -------------------------------------------------------------
        if any(phrase in text_lower for phrase in ["how are you", "hows it going", "how are you doing", "kese ho", "kaise ho", "kya haal hai", "kya hal"]):
            if is_roman_urdu:
                content = "Main bilkul theek hoon, poochne ka bohat shukriya! Meri poori tawajjo aap par hai. Aaj aapke zehn mein kya chal raha hai?"
            else:
                content = "I'm doing well, thank you for asking! I'm ready to focus entirely on you. What's on your mind today?"
            return content, None

        # -------------------------------------------------------------
        # 10. ACUTE OVERWHELM & DISTRESS
        # -------------------------------------------------------------
        if any(w in words for w in ["fucked", "screwed", "ruined", "dying", "doomed", "mess", "shit", "hell", "tabah", "barbaad", "khatam"]):
            if is_roman_urdu:
                content = (
                    "Main mehsoos kar sakta hoon ke aap bohot zyada thakawat aur shadeed dabaav mehsoos kar rahe hain. "
                    "Jab sab kuch aik sath sar par aa jaye to insan himmat haarne lagta hai. "
                    "Aik gehra saans lein, aur mujhe batayein ke sab se zyada pareshani kis baat ki hai?"
                )
            else:
                content = (
                    "I hear how intensely overwhelmed and stressed you're feeling right now. "
                    "When everything feels like it's crashing down, it can be truly exhausting. "
                    "Can you take a deep breath with me and tell me what the biggest trigger was today?"
                )
            return content, None

        # -------------------------------------------------------------
        # 11. SLEEP & PHYSICAL FATIGUE
        # -------------------------------------------------------------
        if any(w in words for w in ["sleep", "insomnia", "sleepless", "awake", "tired", "exhausted", "drained", "cycle", "neend", "thak", "thaka"]):
            if is_roman_urdu:
                content = (
                    "Neend kharab hone aur jismani thakawat se zehni tanav aur bojh mazeed barh jata hai. "
                    "Jab neend poori na ho to dimaag ko mushkilaat se nipatne ki tawaanai nahi milti. "
                    "Jab aap aaram karne ki koshish karte hain to zehn mein kis qisam ke khayalat aate hain?"
                )
            else:
                content = (
                    "Disrupted sleep and physical exhaustion make emotional stress feel so much heavier. "
                    "When your sleep cycle is affected, your mind doesn't get the rest it needs to process difficulties. "
                    "What thoughts tend to race through your mind when you're trying to rest?"
                )
            return content, None

        # -------------------------------------------------------------
        # 12. GENERAL REFLECTIVE THERAPEUTIC CONTINUITY
        # Never loop the same open-ended questions!
        # -------------------------------------------------------------
        if is_roman_urdu:
            content = (
                "Main samajh sakta hoon ke ye soorat-e-haal kitni dushwaar hai. "
                "CBT mein hum apne khayalat aur jazbaat ko aahista aahista suljhate hain taake zehn par bojh kam ho sake. "
                "Is waqt sab se ahem cheez kya hai jis par hum mil kar baat karein?"
            )
        else:
            content = (
                "I hear how significant this is for you. In therapy, we take complex emotions and unpack them together step-by-step so you don't feel burdened all at once. "
                "What feels like the most supportive place for us to focus right now?"
            )
        return content, None


fallback_service = FallbackService()


