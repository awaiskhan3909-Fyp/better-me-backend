import re
from typing import Tuple, Optional, Dict, Any
from app.schemas.response import CBTGuidance, DistortionPrediction


class FallbackService:
    """
    Deterministic Fallback Response Service used when LLM API is unavailable,
    times out, or generates invalid responses.
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

    CBT_ROMAN_URDU_REFRAMES = {
        "Overgeneralization": (
            "Aisa lagta hai ke aik waqiye ki wajah se aapko lag raha hai ke sab kuch hi kharab ho gaya hai (Overgeneralization). "
            "CBT ke mutabiq ye aik fikri khata hai. Kya hum mil kar dekh sakte hain ke kya koi aisi cheez hai jo abhi bhi theek chal rahi hai?"
        ),
        "Catastrophizing": (
            "Aapka zehn foran badtareen soorat-e-haal ki taraf ja raha hai (Catastrophizing). "
            "Jab hum pareshan hote hain to zehan har cheez ko tabahi samajhta hai. "
            "Agar hum thehar kar thandi dimaag se sochein, to haqeeqat mein sab se zyada imkaan kis baat ka hai?"
        ),
        "All-or-Nothing Thinking": (
            "Aap cheezon ko sirf 'poori kamyaabi ya mukammal nakami' ke do pehlo mein dekh rahe hain (All-or-Nothing Thinking). "
            "Zindagi mein darmiyani raste aur seekhne ke marhale bhi hotay hain. Aap is baare mein kya sochte hain?"
        ),
        "Emotional Reasoning": (
            "Aapko aisa mehsoos ho raha hai, is liye lag raha hai ke haqeeqat bhi aisi hi hai (Emotional Reasoning). "
            "Lekin jazbaat aur haqeeqat do alag cheezein hain. Kya aap is ehsas ka koi thos saboot dekh sakte hain?"
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

    ROTATING_EXPLORATORY_PROMPTS = [
        "I'm listening and here with you. What specific moment or thought has been feeling the most difficult today?",
        "Thank you for sharing that with me. Could you walk me through what's been happening around this situation?",
        "It takes courage to open up about these things. When you notice these feelings coming up, what thoughts usually accompany them?",
        "I want to make sure I understand what you're experiencing. How long have you been carrying this stress?",
        "I hear how much this is impacting you. If we were to unpack this one step at a time, where would you like to start?"
    ]

    ROTATING_EXPLORATORY_ROMAN_URDU_PROMPTS = [
        "Main aapki baat ghaur se sun raha hoon. Is soorat-e-haal ke baare mein thora mazeed batayein taake hum mil kar iska hal talaash karein.",
        "Apne dil ki baat kehna hosle ki alamat hai. Jab ye khayal aata hai to aap kaisa mehsoos karte hain?",
        "Main yahan aapke sath hoon. Agar hum isko aahista aahista samjhein, to aap kahan se shuru karna chahenge?",
        "Aapki baat bilkul samajh aa rahi hai. Is waqt sab se zyada konsi baat aapke zehn par bojh daal rahi hai?"
    ]

    def generate_fallback(
        self,
        user_text: str,
        strategy: str,
        cbt_guidance: Optional[CBTGuidance] = None,
        is_roman_urdu: bool = False
    ) -> Tuple[str, Optional[Dict[str, Any]]]:

        if strategy == "safety_response":
            if is_roman_urdu:
                return self.SAFETY_OVERRIDE_ROMAN_URDU_TEXT, None
            return self.SAFETY_OVERRIDE_TEXT, None

        elif strategy == "cbt_support":
            distortion_name = cbt_guidance.detected_distortion if cbt_guidance else None
            if is_roman_urdu:
                reframe = self.CBT_ROMAN_URDU_REFRAMES.get(
                    distortion_name,
                    "Main samajh sakta hoon ke ye khayal aapke liye kitna mushkil aur pareshan kun hai. "
                    "CBT mein hum apne khayalat ko haqeeqat ke tarazu mein toltay hain. "
                    "Kya is baat ko kisi aur mutawazan (balanced) tarha se bhi dekha ja sakta hai?"
                )
                cbt_dump = cbt_guidance.model_dump() if cbt_guidance else {}
                cbt_dump["balanced_thought_guidance"] = reframe
                return reframe, cbt_dump
            elif cbt_guidance and cbt_guidance.balanced_thought_guidance:
                return cbt_guidance.balanced_thought_guidance, cbt_guidance.model_dump()

        text_lower = user_text.lower()
        words = set(re.findall(r"\b[a-zA-Z']+\b", text_lower))

        # 1. Identity & Role Queries
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

        # 2. Greetings
        if any(w in words for w in ["hi", "hello", "hey", "assalam", "salam"]):
            if is_roman_urdu:
                content = "Salam! Main yahan aapki baat sunne aur aapko support karne ke liye tayyar hoon. Aap kaisa mehsoos kar rahe hain, aur aaj kis baare mein baat karna chahenge?"
            else:
                content = "Hello! I'm here to listen and help you work through any thoughts or situations on your mind today. How are you feeling right now?"
            return content, None

        # 3. How are you
        if any(phrase in text_lower for phrase in ["how are you", "hows it going", "how are you doing", "kese ho", "kaise ho", "kya haal hai", "kya hal"]):
            if is_roman_urdu:
                content = "Main bilkul theek hoon, poochne ka bohat shukriya! Meri poori tawajjo aap par hai. Aaj aapke zehn mein kya chal raha hai?"
            else:
                content = "I'm doing well, thank you for asking! I'm ready to focus entirely on you. What's on your mind today?"
            return content, None

        # 4. Acute Overwhelm / Crisis Slang
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

        # 5. Sleep, Insomnia & Physical Fatigue
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

        # 6. Helplessness & Hopelessness
        if any(w in words for w in ["helpless", "hopeless", "useless", "worthless", "stuck", "pointless", "alone", "lonely", "mayoos", "akela", "tanha", "bekaar"]):
            if is_roman_urdu:
                content = (
                    "Mayoosi aur akelepan ka ehsas bohot takleef-deh hota hai, aur aapka thak jana samajh aata hai. "
                    "Lekin yaad rakhein ke jab hum pareshan hote hain to hamara zehn cheezon ko haqeeqat se zyada tareek dikhata hai. "
                    "Aap is safar mein akelay nahi hain, is waqt sab se bhaari ehsas kya hai?"
                )
            else:
                content = (
                    "Feeling helpless is a really painful place to be, and it's completely understandable that you feel drained. "
                    "Even when our thoughts tell us nothing can change, that is often distress talking rather than the whole reality. "
                    "What feels like the heaviest part of this situation right now?"
                )
            return content, None

        # 7. Academic / Exam Stress
        if any(w in words for w in ["exam", "test", "study", "studies", "result", "grade", "college", "school", "university", "paper", "failed", "marks", "parhai", "fail", "imtihan"]):
            if is_roman_urdu:
                content = (
                    "Imtihan ya parhai mein mushkil aana bohot discouraging aur pareshan kun hota hai. "
                    "Lekin yaad rakhein ke aik exam aapki qabiliyat ya poore mustaqbil ka faisla nahi karta. "
                    "CBT mein hum isko aik waqti challenge ke tor par dekhte hain. Is waqt aapke zehn mein kya khayalat aa rahe hain?"
                )
            else:
                content = "Having a tough time with an exam or studies can feel really discouraging and stressful. What thoughts have been running through your mind since this happened?"
            return content, None

        # 8. Interpersonal / Relationship Issues (Strict word boundaries)
        if any(w in words for w in ["friend", "father", "mother", "brother", "sister", "mom", "dad", "partner", "boyfriend", "girlfriend", "boss", "colleague", "dost", "ammi", "abbu", "bhai", "behen", "rishta"]):
            if is_roman_urdu:
                content = "Aisa lagta hai ke kisi qareebi rishte ya baat-cheet ki wajah se pareshani ho rahi hai. Kya aap batana chahenge ke kya hua aur iska aap par kya asar para?"
            else:
                content = "It sounds like relationship dynamics or concerns about someone close to you are playing a big role. What has been happening between you two, and how is it affecting you?"
            return content, None

        # 9. Work & Career Stress
        if any(w in words for w in ["work", "job", "career", "office", "interview", "salary", "fired", "promotion", "naukri", "kaam", "mulazmat"]):
            if is_roman_urdu:
                content = "Naukri ya career ke masayel waqatan zehni dabaav ka baais bantay hain. Is waqt kis cheez ne sab se zyada pareshan kiya hua hai?"
            else:
                content = "Dealing with challenges at work or career can feel really demanding. Could you tell me what specific part of this situation is causing the most stress?"
            return content, None

        # 10. General Exploratory (Rotating to prevent repetition)
        if is_roman_urdu:
            prompt_idx = abs(hash(user_text)) % len(self.ROTATING_EXPLORATORY_ROMAN_URDU_PROMPTS)
            content = self.ROTATING_EXPLORATORY_ROMAN_URDU_PROMPTS[prompt_idx]
        else:
            prompt_idx = abs(hash(user_text)) % len(self.ROTATING_EXPLORATORY_PROMPTS)
            content = self.ROTATING_EXPLORATORY_PROMPTS[prompt_idx]
        return content, None


fallback_service = FallbackService()

