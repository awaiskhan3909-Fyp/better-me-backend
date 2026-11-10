import re
import random
from typing import Tuple, Optional, Dict, Any, List
from app.schemas.response import CBTGuidance, DistortionPrediction
from app.services.distortion_service import CANONICAL_DISTORTIONS


class FallbackService:
    """
    Advanced Clinical CBT Multi-Stage Failover Engine (Beckian Protocol & 5-Part Model):
    - Strictly adheres to Single Question Rule.
    - Context-aware question deduplication: filters out questions already asked.
    - Handles confusion, frustration, off-topic replies, and direct questions gracefully.
    - Multi-stage depth: Situation -> Somatic/Emotion -> Automatic Thoughts -> Distortion -> Restructuring -> Action -> Structured Closure.
    - Bilingual (English & conversational Roman Urdu).
    - NEVER leaks raw 'LABEL_' strings into outputs.
    """

    SAFETY_OVERRIDE_TEXT = (
        "Your safety and well-being are the absolute priority. "
        "If you or someone you know is experiencing thoughts of self-harm, suicide, or overwhelming crisis, "
        "please reach out immediately to a trusted healthcare professional or emergency helpline (such as 988).\n"
        "Support is available 24/7 and you do not have to carry this alone."
    )

    SAFETY_OVERRIDE_ROMAN_URDU_TEXT = (
        "Aapki hifazat aur zindagi bohot qeemti hai. "
        "Agar aap shadeed takleef ya khudkushi ke khayalat se guzar rahe hain, "
        "to barah-e-karam foran kisi doctor ya crisis helpline se rabta karein:\n"
        "• Pakistan Mental Health Helpline (Umang): 0311-7786264 / 0317-4288898\n"
        "• International Crisis Helpline: 988\n"
        "Aap akele nahi hain, madad 24/7 dastiyab hai."
    )

    META_RESPONSES = {
        "identity_and_role_query": {
            "eng": "I am Better Me, an AI-based clinical companion grounded in Cognitive Behavioral Therapy (CBT). My role is to help you explore challenging situations, understand automatic thoughts, and discover balanced, resilient ways forward. What would feel most helpful for us to focus on today?",
            "urdu": "Main Better Me AI hoon, ek CBT-grounded clinical companion. Mera maqsad aapko mushkil halaat ko samajhne aur zehni dabao ko kam karne mein madad dena hai. Aaj aap kis maslay par baat karna chahenge?"
        },
        "user_confusion_clarification": {
            "eng": "I want to be completely clear and straightforward with you. In simple terms, we can look at the situation together step by step, without any complex terms or rushed conclusions. What part felt unclear or confusing to you?",
            "urdu": "Main bilkul aasan alfaz mein wazeh karna chahta hoon taake koi uljhan na rahe. Hum baghair kisi mushkil istilaah ke aaram se baat kar sakte hain. Kaun si baat samajh nahi aayi thi?"
        },
        "user_negative_feedback": {
            "eng": "Thank you for being open and telling me that directly. I really hear your frustration, and it is exhausting when questions feel repetitive. Let's pause the questions—would you prefer if I share a direct summary of what you've shared so far, or is there a specific step you want to take right now?",
            "urdu": "Aapka sach batane ka bohot shukriya. Main samajh sakta hoon ke bar bar sawalat sunna kitna thaka dene wala hota hai. Aayein sawal rokte hain—kya aap chahenge ke main ab tak ki baat ka khulasa karun ya aap kisi khaas cheez par baat karna chahte hain?"
        },
        "off_topic_bridge": {
            "eng": "That sounds nice! Taking a brief mental pause can be refreshing. Whenever you're ready, coming back to what we were exploring earlier about your situation, what has been on your mind?",
            "urdu": "Zabardast! Kabhi kabhi zehan ko halka karne ke liye aisi baat karna acha hota hai. Jab bhi aap tayyar hon, jo masla hum pehle discuss kar rahe thay, us hawale se is waqt aap kya mehsoos kar rahe hain?"
        },
        "direct_question_answer": {
            "eng": "That is an important question. In CBT, the goal is not to force artificial positivity, but to test whether our worst fears are true facts or anxious assumptions. What feels like the biggest obstacle standing in your way right now?",
            "urdu": "Yeh bohot ahem sawal hai. CBT ka maqsad zabardasti 'positive' sochna nahi hota, balkay yeh dekhna hota hai ke kya hamara khauf waqie sach hai ya sirf pareshani. Is waqt aapko sab se bari rukawat kya lag rahi hai?"
        },
        "mid_session_greeting_acknowledgment": {
            "eng": "I'm right here with you. We were in the middle of talking through what happened—how are you feeling about it right now?",
            "urdu": "Main bilkul yahan aapke sath hoon. Hum aapke maslay par baat kar rahe thay—is waqt aapke dil mein kya chal raha hai?"
        },
        "polite_acknowledgment_continue": {
            "eng": "You are very welcome. I'm really glad we can explore this together safely. To help us understand the full picture, how has this been affecting your daily routine or sleep?",
            "urdu": "Aapka bohot shukriya. Mujhe khushi hai ke hum mil kar is par baat kar rahe hain. Is baat ko mazeed samajhne ke liye, kya is pareshani ka asar aapki neend ya rozmarrah ke kaamon par par raha hai?"
        },
        "perfectionism_exploration": {
            "eng": "Holding yourself to an unrelenting standard of 100% perfection can be emotionally exhausting. What is the emotional cost to your peace of mind when you demand that you must never make a mistake?",
            "urdu": "Har cheez ke 100% perfect hone ka mutalba zehan par bohot shadeed bojh dalta hai. Is soch ki wajah se aapko kitna zehni dabao aur thakan mehsoos hoti hai?"
        },
        "uncertainty_grounding": {
            "eng": "It is completely okay not to be certain or to feel unsure right now. There is zero pressure to have all the answers—what is the first small thought that comes to mind when you look at this situation?",
            "urdu": "Bilkul koi baat nahi agar is waqt sab kuch wazeh na ho ya aap unsure hon. Zehan par koi dabao mat daalein—jab aap is baat ke baare mein sochte hain to dil mein sab se pehla khayal kya aata hai?"
        }
    }

    STAGE_POOLS_ENG = {
        "check_in": [
            "Hello! I am Better Me, your supportive CBT AI companion. How are you feeling right now, and what is on your mind today?",
            "Welcome! I am here to support you in a calm, non-judgmental space. How has your emotional state been feeling today?",
            "Hi there. I'm glad you're here. What would feel most important for us to unpack together today?"
        ],
        "exploration": [
            "I hear you. To help us understand the full context, what specific situation or event triggered this feeling recently?",
            "Thank you for sharing that with me. Could you walk me through what was happening right around the time this started feeling heavy?",
            "It makes complete sense that this is weighing on you. Who was involved, and what specifically took place when this occurred?",
            "Understanding the concrete details helps a lot. When this happened, where were you and what set it off?"
        ],
        "emotion_and_body": [
            "When this situation happened, what emotions felt the strongest—like anxiety, sadness, anger, or feeling overwhelmed?",
            "Our bodies often carry distress before our minds fully process it. Did you notice physical reactions like tension, a racing heart, a pit in your stomach, or trouble sleeping?",
            "If you were to rate how intense this distress felt on a scale of 0 to 100, where would it be?",
            "How has this emotional weight affected your day-to-day actions—such as avoiding work, isolating from others, or finding it hard to focus?"
        ],
        "thought_identification": [
            "In that exact moment when the distress was at its peak, what was the primary automatic thought running through your mind?",
            "If you could put words to the exact belief or assumption you had about yourself or the situation right then, what was it?",
            "Often our minds jump to a harsh conclusion when upset. What specific thought were you telling yourself?"
        ],
        "distortion_analysis": {
            "Catastrophizing": [
                "When distress peaks, our minds often jump straight to the absolute worst-case disaster. Looking at the facts realistically, what is one realistic outcome that lies between the worst-case scenario and what you are picturing?",
                "Predicting that your entire future is ruined is often catastrophic thinking rather than reality. What is one piece of concrete evidence showing you have survived and handled tough setbacks before?"
            ],
            "Fortune Telling": [
                "It sounds like anxiety is predicting a painful failure as if it were a guaranteed certainty. What verified facts prove that the future outcome is already set in stone?",
                "Predicting failure ahead of time is a common protective response. What is one manageable possibility that you might not be considering right now?"
            ],
            "Overgeneralization": [
                "Experiencing a setback in this one instance is painful, but does a single event mean you are destined to fail every time? Can you recall a time when an initial setback did not stop you from eventually succeeding?",
                "It is easy to believe 'nothing ever works' after a disappointment. What is one specific lesson or positive effort you can acknowledge from what happened?"
            ],
            "Emotional Reasoning": [
                "I hear how intense these feelings are, and they are completely valid. However, feeling like something is hopeless does not make it an objective reality. If an impartial observer looked at the facts of your situation, what would they say?",
                "Strong emotions can make our worries feel like undeniable truths. What is one concrete fact about your current abilities that remains solid regardless of how upset you feel right now?"
            ],
            "Mind Reading": [
                "It is exhausting when our minds tell us that others are judging us or looking down on us. What verified evidence do you have that proves what other people are actually thinking about you?",
                "Assuming negative reactions from others often isolates us from those who care. Is it possible that someone around you wants to support you, but simply does not realize how much you are hurting inside?"
            ],
            "All-or-Nothing Thinking": [
                "Viewing things in black-and-white (either complete success or total failure) ignores all the effort and learning in between. What would a more balanced, middle-ground view of this situation look like?",
                "Life is rarely all-or-nothing; experiencing a setback does not erase your worth or your hard work. What is one partial success or valuable lesson you can acknowledge from what happened?"
            ],
            "General": [
                "Let's look at this thought together with curiosity rather than self-criticism. What factual evidence supports this belief, and what evidence might point to another perspective?",
                "Sometimes our immediate thoughts present only part of the story. If a close friend were in your exact shoes right now, what perspective would you offer them?"
            ]
        },
        "intervention": [
            "Considering all the facts we've explored, how could we reframe this in a more balanced, realistic way that supports your growth?",
            "If we step back from harsh self-criticism, what is a kinder and more accurate statement you can tell yourself today?",
            "What is a constructive perspective that acknowledges the challenge without treating it as an impossible roadblock?"
        ],
        "action_plan": [
            "We have discovered a much more balanced perspective today. What is one small, manageable micro-step you can take today—like trying a 2-minute breathing exercise, taking a short walk, or writing down your next task?",
            "Turning reflection into action can help ease the physical weight. Would you be open to trying a brief grounding exercise or writing down this balanced thought where you can see it?",
            "What is one gentle, actionable thing you can do for yourself over the next couple of hours to protect your peace of mind?"
        ],
        "closure": [
            (
                "--- Session Summary & Takeaway ---\n"
                "• Core Reflection: You noticed how easy it is for overwhelming stress to trigger catastrophic assumptions, and we uncovered that one setback does not define your ongoing capabilities.\n"
                "• Personalized Homework: Try writing down your balanced perspective on an index card, and take one 2-minute slow breathing pause whenever tension rises.\n"
                "• Next Session Goal: We will check in on how that behavioral micro-step felt and continue building your cognitive resilience. Take gentle care of yourself today!"
            ),
            (
                "--- Session Summary & Takeaway ---\n"
                "• Core Reflection: You examined the evidence with courage and discovered that your automatic thought was an anxious prediction rather than a permanent fact.\n"
                "• Personalized Homework: Practice noticing when 'all-or-nothing' thoughts pop up, and remind yourself of the middle ground.\n"
                "• Next Session Goal: In our next session, we can review how this balanced mindset supported your daily routine. I am always here whenever you want to check in!"
            )
        ]
    }

    STAGE_POOLS_URDU = {
        "check_in": [
            "Salam! Main Better Me AI hoon, aapka supportive clinical companion. Aap is waqt kaisa mehsoos kar rahe hain, aur aaj kis baare mein baat karna chahenge?",
            "Khush-amdeed! Main yahan aapke sath ek pursukoon mahol mein baat karne ke liye maujood hoon. Aaj aapka din kaisa guzar raha hai?",
            "Salam! Aaj aapke zehan par kaun si aisi baat hai jis par hum baat kar sakte hain?"
        ],
        "exploration": [
            "Main samajh sakta hoon. Is baat ko achi tarah samajhne ke liye, kya aap bata sakte hain ke kis khaas waqiye ya baat se yeh ehsas shuru hua?",
            "Aapka yeh share karne ka shukriya. Asal mein kya hua tha jab aapne aisa sochna shuru kiya?",
            "Is pareshani ke piche kya soorat-e-haal thi? Kahan aur kab yeh sab shuru hua?"
        ],
        "emotion_and_body": [
            "Jab yeh sab hua, to dil mein sab se zyada kaun sa ehsas tha—khauf, udasi, gussa ya ghabrahat?",
            "Aksar pareshani jism par bhi asar dalti hai. Kya aapne seene mein khichao, dil ki tez dhadkan, pait mein dard ya neend ki kami mehsoos ki?",
            "Agar aap is takleef ki shiddat ko 0 se 100 ke darmiyan number dein, to yeh kitni shadeed thi?"
        ],
        "thought_identification": [
            "Jab yeh takleef apne urooj par thi, to aapke zehan mein sab se pehla automatic khayal kya aaya tha?",
            "Agar aap us lamhe ki soch ko lafzon mein bayan karein, to aapne apne baare mein kya socha tha?",
            "Is waqiye ke foran baad aapke dil ne kya faisla sunaya?"
        ],
        "distortion_analysis": {
            "Catastrophizing": [
                "Jab pareshani zyada hoti hai to dimagh aksar sab se buri soorat ke baare mein sochne lagta hai. Haqeeqat ko dekhte hue, aisi kaun si darmiyani soorat hai jo is intehaai khof se behtar ho?",
                "Kisi aik maslay ka matlab yeh hargiz nahi hota ke poora mustaqbil tabah ho gaya hai. Kya maazi mein koi aisi mushkil aayi thi jise aapne sabr se hal kar liya tha?"
            ],
            "Fortune Telling": [
                "Mustaqbil ke baare mein pehle se bura nateeja nikal lena hamari pareshani ka nateeja hota hai, haqeeqat ka nahi. Kaun sa thos saboot hai jo yeh sabit kare ke aane wala kal bura hi hoga?",
                "Aane wale waqt ki itni fikr hona samajh aata hai. Is waqt kaun sa chota sa qadam uthaya ja sakta hai jo aane wale kal ko behtar bana sake?"
            ],
            "Overgeneralization": [
                "Aik baar ki mushkil ya nakami ka matlab yeh nahi ke aap hamesha nakam hi rahenge. Kya maazi mein kabhi aisa hua ke shuruati mushkil ke baad aapne kamyabi haasil ki ho?",
                "Yeh sochna ke 'sab kuch bekaar hai' aik jhatke ka asar hota hai. Is tajurbay se aisi kaun si aik baat seekhne ko mili jo aage kaam aa sake?"
            ],
            "Emotional Reasoning": [
                "Aapke jazbaat bohot shadeed hain aur is waqt udas hona bilkul samajh aata hai. Lekin sirf bura mehsoos karne ka matlab yeh nahi hota ke sab kuch waqie bura ho gaya hai. Haqeeqat ke aaine mein yeh kaisa dikhta hai?",
                "Shadeed jazbaat aksar haqeeqat ko dhundla dete hain. Aapki aisi kaun si qabliyat hai jo is mushkil ke bawajood bilkul qaim hai?"
            ],
            "Mind Reading": [
                "Yeh bohot thaka dene wali baat hoti hai jab hamara zehan doosron ke khayalat ka andaza lagata hai. Kya aapke paas koi pakka saboot hai ke doosray log waqie wahi soch rahe hain jo aap samajh rahe hain?",
                "Aksar hum doosron ke chehre par apni pareshani dekhne lagte hain. Kya yeh mumkin hai ke koi aapki madad karna chahta ho lekin use aapki takleef ka pata na ho?"
            ],
            "All-or-Nothing Thinking": [
                "Zindagi ya to 'mukammal kamyabi' ya 'poori nakami' nahi hoti; iske darmiyan bhi bohot se darjajaat hote hain. Iska darmiyani rasta kya ho sakta hai jo aapki koshish ko tasleem kare?",
                "Har cheez ka perfect hona zaroori nahi hota. Is koshish mein aisi kaun si baat thi jo aapne achi ki, bhale nateeja man chaha na raha ho?"
            ],
            "General": [
                "Aayein mil kar is soch ka jaiza lete hain. Kaun sa saboot is soch ki himayat karta hai, aur kaun sa saboot iske mukhalif hai?",
                "Agar aapka koi azeez dost is mushkil mein hota, to aap usay kya mashwara dete?"
            ]
        },
        "intervention": [
            "In sab baton ko samne rakhte hue, is surat-haal ko dekhne ka zyada mutawazan aur behtar andaz kya ho sakta hai?",
            "Agar hum is shadeed tanqeed se thora peeche hatein, to aap apne aap ko kya tasalli-bakhsh aur haqeeqat-pasandana baat keh sakte hain?",
            "Kaun si aisi nayi soch hai jo aapko is mushkil ke bawajood agay barhne ki umeed de sakti hai?"
        ],
        "action_plan": [
            "Aaj humne aik behtar soch tak pohanchne ki koshish ki hai. Ab agay ke liye, kya aap koi chota sa aasan qadam utha sakte hain—jaise 2 minute gehri saans lena ya thori der taaza hawa mein tehelna?",
            "Is baat ko aasan banane ke liye, aaj ke din ke liye kaun sa chota sa amli kaam aapke liye qabil-e-amal hoga?",
            "Apne zehan ko sakoon dene ke liye, agle aik ghante mein aap apne liye kya behtar kar sakte hain?"
        ],
        "closure": [
            (
                "--- Session Ka Khulasa Aur Homework ---\n"
                "• Aham Nateeja: Aaj humne dekha ke pareshani mein hamara zehan sab se bura sochne lagta hai, lekin aik nakami poori zindagi ka faisla nahi hoti.\n"
                "• Homework: Jab bhi zehan mein 'sab khatam' ka khayal aaye, to is mutawazan soch ko yaad karein aur 2 minute gehri saans lein.\n"
                "• Agla Maqsad: Aglay session mein hum dekhenge ke is chotay amli qadam se aapko kitna faida hua. Apna bohot khayal rakhein!"
            )
        ]
    }

    def _select_non_duplicate(self, candidate_list: List[str], asked_questions: List[str]) -> str:
        """Selects a candidate that has not already been asked."""
        if not candidate_list:
            return ""
        if not asked_questions:
            return random.choice(candidate_list)

        clean_asked = [re.sub(r'[^\w\s]', '', q.lower()).strip() for q in asked_questions]
        unasked = []
        for cand in candidate_list:
            clean_cand = re.sub(r'[^\w\s]', '', cand.lower()).strip()
            if not any(clean_cand in q or q in clean_cand for q in clean_asked):
                unasked.append(cand)

        if unasked:
            return random.choice(unasked)
        return random.choice(candidate_list)

    def generate_fallback(
        self,
        user_text: str,
        strategy: str,
        decision_reason: str = "",
        therapy_stage: Optional[str] = None,
        cbt_guidance: Optional[CBTGuidance] = None,
        conversation_context: Optional[Any] = None,
        distortion_pred: Optional[DistortionPrediction] = None,
        asked_questions: Optional[List[str]] = None,
        is_roman_urdu: bool = False
    ) -> Tuple[str, Optional[Dict[str, Any]]]:

        asked_q = asked_questions or []

        # 1. STRICT SAFETY CRISIS OVERRIDE GATE
        if strategy == "safety_response" or decision_reason == "high_risk_safety_alert":
            if is_roman_urdu:
                return self.SAFETY_OVERRIDE_ROMAN_URDU_TEXT, None
            return self.SAFETY_OVERRIDE_TEXT, None

        # 2. DIRECT META QUERIES, CONFUSION, FRUSTRATION & BYPASSES
        if decision_reason in self.META_RESPONSES:
            lang_key = "urdu" if is_roman_urdu else "eng"
            return self.META_RESPONSES[decision_reason][lang_key], None

        # 3. RESOLVE CANONICAL STAGE
        stage = therapy_stage or "exploration"
        if stage in ["problem_exploration", "exploration"]:
            stage = "exploration"
        elif stage == "emotion_and_body":
            stage = "emotion_and_body"
        elif stage in ["thought_identification", "assessment"]:
            stage = "thought_identification"
        elif stage in ["socratic_intervention", "intervention"]:
            stage = "intervention"

        # 4. RESOLVE SANITIZED DISTORTION NAME (NEVER LABEL_X)
        detected_distortion = None
        if distortion_pred and distortion_pred.predicted_class not in ["None", "Normal", "Unknown", None]:
            detected_distortion = distortion_pred.predicted_class
        elif cbt_guidance and cbt_guidance.detected_distortion not in ["None", "Normal", "Unknown", "General", None]:
            detected_distortion = cbt_guidance.detected_distortion

        if str(detected_distortion).startswith("LABEL_"):
            try:
                idx = int(str(detected_distortion).split("_")[-1])
                detected_distortion = CANONICAL_DISTORTIONS.get(idx, "None")
            except Exception:
                detected_distortion = "None"

        response_text = ""
        cbt_data = None

        # 5. DISTORTION ANALYSIS STAGE (Stage 5)
        if stage == "distortion_analysis":
            pool_dict = self.STAGE_POOLS_URDU["distortion_analysis"] if is_roman_urdu else self.STAGE_POOLS_ENG["distortion_analysis"]
            distortion_options = pool_dict.get(detected_distortion, pool_dict["General"])
            response_text = self._select_non_duplicate(distortion_options, asked_q)

            cbt_data = {
                "detected_distortion": detected_distortion or "General",
                "title": f"Cognitive Restructuring: {detected_distortion or 'Balanced Perspective'}",
                "balanced_thought_guidance": response_text
            }

        # 6. ALL OTHER CANONICAL STAGES
        elif stage in ["check_in", "exploration", "emotion_and_body", "thought_identification", "intervention", "action_plan", "closure"]:
            pool_list = (self.STAGE_POOLS_URDU if is_roman_urdu else self.STAGE_POOLS_ENG).get(stage, [])
            if pool_list:
                response_text = self._select_non_duplicate(pool_list, asked_q)
            else:
                response_text = (
                    "Main aapki baat ko tawajjoh se sun raha hoon. Is hawale se mazeed kya hua tha?"
                    if is_roman_urdu else
                    "I am listening closely to what you are sharing. Could you tell me more about how that unfolded?"
                )
        else:
            response_text = (
                "Main aapki baat samajh sakta hoon. Aayein isay step-by-step dekhein; is waqt sab se zaroori baat kya hai?"
                if is_roman_urdu else
                "I hear you clearly. Let's unpack this together step by step; what feels most important to focus on right now?"
            )

        if cbt_guidance and not str(cbt_guidance.detected_distortion).startswith("LABEL_"):
            cbt_data = cbt_guidance.model_dump()
            cbt_data["balanced_thought_guidance"] = response_text

        return response_text, cbt_data


fallback_service = FallbackService()
