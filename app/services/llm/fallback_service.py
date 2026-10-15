import re
import random
from typing import Tuple, Optional, Dict, Any
from app.schemas.response import CBTGuidance, DistortionPrediction


class FallbackService:
    """
    Evidence-Based Clinical CBT Dynamic Failover Engine (Beckian Cognitive Restructuring).
    Acts as an intelligent failover if GPU inference experiences cold-start or temporary delay.
    Strictly follows clinical CBT rules:
    1. Validates patient's emotional distress first (empathetic connection).
    2. Directly challenges the specific detected Cognitive Distortion.
    3. Concludes with exactly ONE focused, non-interrogative Socratic question.
    4. Provides complete bilingual support (English and natural Roman Urdu).
    """

    SAFETY_OVERRIDE_TEXT = (
        "Your safety and well-being are the top priority. "
        "If you or someone you know is experiencing thoughts of self-harm, suicide, or severe crisis, "
        "please reach out immediately to a trusted professional, doctor, or emergency helpline (such as 988).\n"
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

    # Dynamic Distortion-Specific CBT Formulations (Empathetic Validation + Reframe + 1 Socratic Question)
    CBT_RESTRUCTURING_ENG = {
        "Catastrophizing": [
            (
                "I can hear how overwhelmed and frightened you feel right now. "
                "When distress peaks, our minds often jump straight to the worst-case scenario and treat it as an absolute certainty. "
                "Looking at this realistically, what is one realistic outcome that lies between the worst-case scenario and what you are picturing?"
            ),
            (
                "It makes total sense that this feels like an unbearable disaster at the moment. "
                "However, predicting that your entire life or future is ruined because of this moment is the catastrophic thought speaking, not reality. "
                "What is one piece of concrete evidence that shows you have survived difficult setbacks before?"
            )
        ],
        "Fortune Telling": [
            (
                "I hear the heavy anxiety in your words about what lies ahead. "
                "Our minds sometimes act like a crystal ball, assuming a painful outcome is guaranteed before it even happens. "
                "What is one positive or manageable possibility that you might not be considering right now?"
            )
        ],
        "Overgeneralization": [
            (
                "I am really sorry you are dealing with this disappointing outcome, and it is completely natural to feel disheartened. "
                "Experiencing a painful setback in one instance does not mean you are destined to fail every time. "
                "Can you think of a time in your life when an initial difficulty did not prevent you from succeeding later?"
            ),
            (
                "It sounds like this experience was deeply frustrating for you. "
                "It is easy to believe 'I always fail' or 'nothing ever works,' but a single event does not define your ongoing capabilities. "
                "What is one specific thing you did well or learned through this effort, even if the final result was disappointing?"
            )
        ],
        "Emotional Reasoning": [
            (
                "I hear how intense these painful emotions are, and it makes complete sense that you are hurting. "
                "While your emotions are 100% valid, feeling like something is hopeless does not make it an objective reality. "
                "If you were to step back and look at the actual facts of the situation, what would an impartial observer say?"
            ),
            (
                "It sounds like you are carrying a tremendous amount of sadness right now. "
                "Strong emotions can make us believe our worst thoughts are facts. "
                "What is one fact about your current situation that remains solid regardless of how upset you feel right now?"
            )
        ],
        "Mind Reading": [
            (
                "It is exhausting and isolating when our thoughts tell us that others are judging us or do not care. "
                "We often project our own self-criticism onto others without any real confirmation. "
                "What verified evidence do you have that proves what other people are actually thinking about you?"
            ),
            (
                "I sense how lonely it feels when you believe nobody understands or values you. "
                "Assuming other people's negative reactions often stops us from reaching out to those who genuinely want to help. "
                "Is it possible that someone around you cares, but does not yet realize how much you are hurting inside?"
            )
        ],
        "All-or-Nothing Thinking": [
            (
                "I understand how heartbreaking it feels when things do not turn out exactly the way you hoped. "
                "Seeing situations in black-and-white (either complete success or total failure) dismisses all your hard work and growth in between. "
                "What would a more balanced, middle-ground view of this situation look like?"
            ),
            (
                "It sounds like you are being very hard on yourself right now. "
                "Life is rarely all-or-nothing; experiencing a setback does not wipe away your effort or your worth. "
                "What is one partial success or valuable lesson you can acknowledge from what happened?"
            )
        ]
    }

    CBT_RESTRUCTURING_URDU = {
        "Catastrophizing": [
            (
                "Main samajh sakta hoon ke yeh soch kitni bhaari aur khofnaak mehsoos ho rahi hai. "
                "Lekin kisi aik imtehan ya mushkil ka yeh matlab hargiz nahi ke aapka poora mustaqbil tabah ho gaya hai. "
                "Is shadeed khof ke ilawa, aisi kaun si doosri possibility hai jo aapke liye naya rasta khol sakti hai?"
            )
        ],
        "Fortune Telling": [
            (
                "Aane wale waqt ke baare mein itni shadeed fikr hona samajh aata hai. "
                "Lekin mustaqbil ke baare mein pehle se bura faisla kar lena hamari pareshani ka nateeja hota hai, haqeeqat ka nahi. "
                "Aap is waqt kaun sa chota sa qadam utha sakte hain jo aane wale kal ko behtar bana sake?"
            )
        ],
        "Overgeneralization": [
            (
                "Is nateeje par dil dukhn aur afsos hona bilkul qudrati baat hai. "
                "Lekin aik dafa ki mushkil ya nakami ka matlab yeh nahi ke aap hamesha nakam hi rahenge. "
                "Kya maazi mein kabhi aisa hua hai ke shuruati mushkil ke bawajood aap ne kisi cheez ko sambhal liya ho?"
            )
        ],
        "Emotional Reasoning": [
            (
                "Aapke jazbaat bohot shadeed hain aur is waqt udas hona bilkul samajh aata hai. "
                "Lekin sirf is wajah se ke hum bura mehsoos kar rahe hain, iska matlab yeh nahi hota ke sab kuch waqie bura ho chuka hai. "
                "Kaun sa aisa thos saboot hai jo is soch ke bar-aks haqeeqat ko wazeh karta hai?"
            )
        ],
        "Mind Reading": [
            (
                "Yeh bohot thaka dene wali baat hoti hai jab hamara zehan doosron ke na-maloom khayalat ka andaza lagata hai. "
                "Aksar yeh hamari apni pareshani hoti hai jo doosron ke chehre par nazar aane lagti hai. "
                "Kya aapke paas koi pakka saboot hai ke doosra shakhs waqie wahi soch raha hai jo aap samajh rahe hain?"
            )
        ],
        "All-or-Nothing Thinking": [
            (
                "Main samajh sakta hoon ke jab mehnat ke bawajood nateeja man chaha na aaye to kitni mayoosi hoti hai. "
                "Lekin zindagi ya to 'mukammal kamyabi' ya 'poori nakami' nahi hoti, iske darmiyan bhi bohot se darjajaat hote hain. "
                "Is surat-haal ka darmiyani rasta kya ho sakta hai jo aapki koshish aur mehnat ko bhi tasleem kare?"
            )
        ]
    }

    def generate_fallback(
        self,
        user_text: str,
        strategy: str,
        cbt_guidance: Optional[CBTGuidance] = None,
        conversation_context: Optional[Any] = None,
        distortion_pred: Optional[DistortionPrediction] = None,
        is_roman_urdu: bool = False
    ) -> Tuple[str, Optional[Dict[str, Any]]]:

        # 1. STRICT SAFETY CRISIS OVERRIDE GATE
        if strategy == "safety_response":
            if is_roman_urdu:
                return self.SAFETY_OVERRIDE_ROMAN_URDU_TEXT, None
            return self.SAFETY_OVERRIDE_TEXT, None

        # 2. GREETINGS & CASUAL OPENERS
        text_lower = user_text.lower().strip()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text_lower))
        if any(w in words for w in ["hi", "hello", "hey", "salam", "assalam", "good morning", "good evening"]):
            if is_roman_urdu:
                return "Salam! Main Better Me AI hoon, aapka supportive clinical companion. Aap is waqt kaisa mehsoos kar rahe hain, aur aaj kis baare mein baat karna chahenge?", None
            return "Hello! I am Better Me, your supportive clinical AI companion. How are you feeling right now, and what is on your mind today?", None

        # 3. IDENTIFY DISTORTION SIGNAL FOR TARGETED CBT GUIDANCE
        detected_distortion = None
        if distortion_pred and distortion_pred.predicted_class not in ["None", "Normal", "Unknown", None]:
            detected_distortion = distortion_pred.predicted_class
        elif cbt_guidance and cbt_guidance.detected_distortion not in ["None", "Normal", "Unknown", "General", None]:
            detected_distortion = cbt_guidance.detected_distortion

        # 4. SELECT DISTORTION-SPECIFIC BECKIAN CBT RESTRUCTURE
        if is_roman_urdu:
            urdu_pool = self.CBT_RESTRUCTURING_URDU.get(detected_distortion)
            if urdu_pool:
                response_text = random.choice(urdu_pool)
            else:
                response_text = (
                    "Main aapki takleef aur pareshani ko samajh sakta hoon. Aisi mushkil surat-haal se guzarna aasan nahi hota. "
                    "Aayein mil kar is soch ko step-by-step examine karein; is waqt kaun si baat aapke dil par sab se zyada bojh bani hui hai?"
                )
        else:
            eng_pool = self.CBT_RESTRUCTURING_ENG.get(detected_distortion)
            if eng_pool:
                response_text = random.choice(eng_pool)
            else:
                response_text = (
                    "I hear the distress and emotional weight in what you are experiencing right now. "
                    "Facing these challenges is genuinely difficult, but you do not have to navigate them alone. "
                    "What part of this situation feels like the heaviest burden on your mind right now?"
                )

        # 5. STRUCTURED CBT GUIDANCE METADATA
        cbt_data = None
        if cbt_guidance:
            cbt_data = cbt_guidance.model_dump()
            cbt_data["balanced_thought_guidance"] = response_text
        elif detected_distortion:
            cbt_data = {
                "detected_distortion": detected_distortion,
                "title": f"Cognitive Restructuring: {detected_distortion}",
                "balanced_thought_guidance": response_text
            }

        return response_text, cbt_data


fallback_service = FallbackService()
