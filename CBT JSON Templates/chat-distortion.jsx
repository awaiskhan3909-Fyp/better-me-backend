import { useState, useRef, useEffect } from "react";

const DISTORTION_TYPES = {
  catastrophizing: { label: "🌪 Catastrophizing", color: "#ef4444", bg: "#fef2f2" },
  blackWhite: { label: "⚫ Black & White Thinking", color: "#8b5cf6", bg: "#f5f3ff" },
  mindReading: { label: "🔮 Mind Reading", color: "#f59e0b", bg: "#fffbeb" },
  overgeneralization: { label: "🔁 Overgeneralization", color: "#3b82f6", bg: "#eff6ff" },
  filtering: { label: "🕶 Mental Filter", color: "#10b981", bg: "#ecfdf5" },
  personalization: { label: "👆 Personalization", color: "#ec4899", bg: "#fdf2f8" },
  none: { label: "✓ No distortion detected", color: "#6b7280", bg: "#f9fafb" },
};

function detectDistortion(text) {
  const t = text.toLowerCase();
  if (/always|never|everyone|nobody|everything|nothing|worst|ruined|disaster/.test(t)) return "overgeneralization";
  if (/the end|hopeless|catastrophe|terrible|awful|destroyed|never recover/.test(t)) return "catastrophizing";
  if (/they think|she thinks|he thinks|they hate|probably mad|must be angry/.test(t)) return "mindReading";
  if (/all my fault|i caused|because of me|i ruined/.test(t)) return "personalization";
  if (/only bad|nothing good|can't see|ignore the/.test(t)) return "filtering";
  if (/perfect|either|complete failure|total success/.test(t)) return "blackWhite";
  return "none";
}

const systemPrompt = `You are a compassionate cognitive behavioral therapy assistant. When users share thoughts, respond with empathy, gently challenge cognitive distortions if present, and offer a reframed perspective. Keep responses concise (2-4 sentences).`;

function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom: "12px",
        animation: "fadeSlide 0.3s ease forwards",
      }}
    >
      {!isUser && (
        <div style={{
          width: 32, height: 32, borderRadius: "50%", background: "linear-gradient(135deg, #667eea, #764ba2)",
          display: "flex", alignItems: "center", justifyContent: "center", marginRight: 8, flexShrink: 0,
          fontSize: 14, color: "#fff", fontWeight: 700,
        }}>C</div>
      )}
      <div style={{
        maxWidth: "72%",
        padding: "10px 14px",
        borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
        background: isUser ? "linear-gradient(135deg, #667eea, #764ba2)" : "#fff",
        color: isUser ? "#fff" : "#1a1a2e",
        boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
        fontSize: 14,
        lineHeight: 1.55,
        fontFamily: "'Lora', Georgia, serif",
      }}>
        {msg.content}
      </div>
      {isUser && (
        <div style={{
          width: 32, height: 32, borderRadius: "50%", background: "#e0e7ff",
          display: "flex", alignItems: "center", justifyContent: "center", marginLeft: 8, flexShrink: 0,
          fontSize: 14, color: "#4f46e5", fontWeight: 700,
        }}>U</div>
      )}
    </div>
  );
}

export default function ChatInterface() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hello. I'm here to help you explore your thoughts. What's on your mind today?" }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [distortion, setDistortion] = useState(null);
  const [history, setHistory] = useState([]);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userText = input.trim();
    setInput("");

    const detected = detectDistortion(userText);
    setDistortion(detected);

    const newUserMsg = { role: "user", content: userText };
    const updatedMessages = [...messages, newUserMsg];
    setMessages(updatedMessages);

    const apiHistory = [...history, { role: "user", content: userText }];
    setLoading(true);

    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: systemPrompt,
          messages: apiHistory,
        }),
      });
      const data = await response.json();
      const assistantText = data.content?.map(b => b.text || "").join("") || "I'm here with you.";
      const assistantMsg = { role: "assistant", content: assistantText };
      setMessages(prev => [...prev, assistantMsg]);
      setHistory([...apiHistory, { role: "assistant", content: assistantText }]);
    } catch {
      setMessages(prev => [...prev, { role: "assistant", content: "Something went wrong. Please try again." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const dist = distortion ? DISTORTION_TYPES[distortion] : null;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=DM+Sans:wght@300;400;500&display=swap');
        @keyframes fadeSlide {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        @keyframes distortionReveal {
          from { opacity: 0; transform: translateY(-6px) scale(0.97); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #f0f4ff; }
        textarea:focus { outline: none; }
        textarea { resize: none; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #c7d2fe; border-radius: 4px; }
      `}</style>

      <div style={{
        minHeight: "100vh",
        background: "linear-gradient(160deg, #f0f4ff 0%, #e8edfb 50%, #eef0fb 100%)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "24px",
        fontFamily: "'DM Sans', sans-serif",
      }}>
        <div style={{
          width: "100%",
          maxWidth: 520,
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}>
          {/* Header */}
          <div style={{ textAlign: "center", paddingBottom: 4 }}>
            <div style={{
              display: "inline-flex", alignItems: "center", gap: 8,
              background: "rgba(255,255,255,0.7)", backdropFilter: "blur(12px)",
              borderRadius: 100, padding: "6px 16px",
              border: "1px solid rgba(255,255,255,0.9)",
              boxShadow: "0 1px 12px rgba(102,126,234,0.12)",
            }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#10b981", boxShadow: "0 0 0 2px #d1fae5" }} />
              <span style={{ fontSize: 13, fontWeight: 500, color: "#4f46e5", letterSpacing: "0.02em" }}>
                Cognitive Clarity · AI Companion
              </span>
            </div>
          </div>

          {/* Chat window */}
          <div style={{
            background: "rgba(255,255,255,0.75)",
            backdropFilter: "blur(20px)",
            borderRadius: 24,
            border: "1px solid rgba(255,255,255,0.9)",
            boxShadow: "0 8px 32px rgba(102,126,234,0.12), 0 1px 0 rgba(255,255,255,0.8) inset",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
          }}>
            {/* Messages */}
            <div style={{
              padding: "20px 16px",
              height: 380,
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
            }}>
              {messages.map((m, i) => <Message key={i} msg={m} />)}
              {loading && (
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                  <div style={{
                    width: 32, height: 32, borderRadius: "50%",
                    background: "linear-gradient(135deg, #667eea, #764ba2)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    fontSize: 14, color: "#fff", fontWeight: 700,
                  }}>C</div>
                  <div style={{
                    background: "#fff", borderRadius: "18px 18px 18px 4px",
                    padding: "10px 14px", boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
                    display: "flex", gap: 4, alignItems: "center",
                  }}>
                    {[0, 0.2, 0.4].map((d, i) => (
                      <div key={i} style={{
                        width: 7, height: 7, borderRadius: "50%", background: "#c7d2fe",
                        animation: `pulse 1.2s ease ${d}s infinite`,
                      }} />
                    ))}
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* Distortion badge */}
            {dist && (
              <div style={{
                margin: "0 16px 12px",
                padding: "8px 14px",
                borderRadius: 12,
                background: dist.bg,
                border: `1px solid ${dist.color}22`,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                animation: "distortionReveal 0.3s ease forwards",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{
                    fontSize: 10, fontWeight: 600, letterSpacing: "0.08em",
                    color: "#9ca3af", textTransform: "uppercase",
                  }}>Detected Pattern</div>
                </div>
                <div style={{
                  fontSize: 12, fontWeight: 600, color: dist.color,
                  background: `${dist.color}18`, borderRadius: 100,
                  padding: "3px 10px",
                }}>{dist.label}</div>
              </div>
            )}

            {/* Input area */}
            <div style={{
              padding: "12px 16px 16px",
              borderTop: "1px solid rgba(199,210,254,0.3)",
              display: "flex",
              gap: 10,
              alignItems: "flex-end",
            }}>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Share what's on your mind..."
                rows={1}
                style={{
                  flex: 1,
                  padding: "10px 14px",
                  borderRadius: 14,
                  border: "1.5px solid #e0e7ff",
                  background: "#f8f9ff",
                  fontSize: 14,
                  fontFamily: "'DM Sans', sans-serif",
                  color: "#1a1a2e",
                  lineHeight: 1.5,
                  maxHeight: 120,
                  overflowY: "auto",
                  transition: "border-color 0.2s",
                }}
                onFocus={e => e.target.style.borderColor = "#818cf8"}
                onBlur={e => e.target.style.borderColor = "#e0e7ff"}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                style={{
                  width: 42, height: 42,
                  borderRadius: 14,
                  border: "none",
                  background: (!input.trim() || loading) ? "#e0e7ff" : "linear-gradient(135deg, #667eea, #764ba2)",
                  color: (!input.trim() || loading) ? "#a5b4fc" : "#fff",
                  cursor: (!input.trim() || loading) ? "not-allowed" : "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  flexShrink: 0,
                  transition: "all 0.2s",
                  boxShadow: (!input.trim() || loading) ? "none" : "0 4px 12px rgba(102,126,234,0.35)",
                }}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
          </div>

          {/* Footer hint */}
          <p style={{
            textAlign: "center", fontSize: 11, color: "#9ca3af",
            letterSpacing: "0.02em",
          }}>
            Press <kbd style={{ background: "#e0e7ff", color: "#6366f1", padding: "1px 5px", borderRadius: 4, fontSize: 10 }}>Enter</kbd> to send · <kbd style={{ background: "#e0e7ff", color: "#6366f1", padding: "1px 5px", borderRadius: 4, fontSize: 10 }}>Shift+Enter</kbd> for new line
          </p>
        </div>
      </div>
    </>
  );
}
