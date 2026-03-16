import { useMemo, useState } from "react";

const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const providerOptions = [
  "openai",
  "gemini",
  "anthropic",
  "perplexity",
  "upstage",
  "mistral",
  "groq",
  "cohere",
  "deepseek"
];

const providerLabels = {
  openai: "OpenAI",
  gemini: "Gemini",
  anthropic: "Anthropic",
  perplexity: "Perplexity",
  upstage: "Upstage",
  mistral: "Mistral",
  groq: "Groq",
  cohere: "Cohere",
  deepseek: "DeepSeek"
};

const providerDescriptions = {
  openai: "General-purpose baseline",
  gemini: "Fast multimodal family",
  anthropic: "Safety-heavy reasoning",
  perplexity: "Search-grounded answers",
  upstage: "Korean-friendly model line",
  mistral: "Compact open-weight style",
  groq: "Fast hosted inference",
  cohere: "Command family models",
  deepseek: "Reasoning-leaning models"
};

function parseNdjsonChunk(buffer, onEvent) {
  let remaining = buffer;
  let newlineIndex = remaining.indexOf("\n");
  while (newlineIndex >= 0) {
    const line = remaining.slice(0, newlineIndex).trim();
    remaining = remaining.slice(newlineIndex + 1);
    if (line) {
      onEvent(JSON.parse(line));
    }
    newlineIndex = remaining.indexOf("\n");
  }
  return remaining;
}

function toErrorMessage(payload) {
  if (!payload) {
    return "Unknown request error";
  }
  if (typeof payload === "string") {
    return payload;
  }
  return payload.detail || payload.message || JSON.stringify(payload);
}

function formatStatus(status) {
  if (!status) {
    return "pending";
  }
  if (typeof status === "string") {
    return status;
  }
  const code = status.status ?? "n/a";
  const detail = status.detail ?? "unknown";
  return `${code} · ${detail}`;
}

function isSuccessStatus(status) {
  if (!status) {
    return false;
  }
  if (typeof status === "string") {
    return status.toLowerCase() === "ok";
  }
  const code = Number(status.status);
  return Number.isFinite(code) && code >= 200 && code < 300;
}

function statusLabel(status) {
  if (isSuccessStatus(status)) {
    return "ok";
  }
  if (!status) {
    return "pending";
  }
  return "error";
}

function uniqBy(items, keyBuilder) {
  const seen = new Set();
  return items.filter((item) => {
    const key = keyBuilder(item);
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

export default function App() {
  const [apiBaseUrl, setApiBaseUrl] = useState(DEFAULT_API_BASE);
  const [question, setQuestion] = useState("");
  const [selectedProviders, setSelectedProviders] = useState(providerOptions.slice(0, 3));
  const [events, setEvents] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [requestError, setRequestError] = useState(null);

  const summary = useMemo(() => {
    for (let index = events.length - 1; index >= 0; index -= 1) {
      if (events[index].event === "summary") {
        return events[index];
      }
    }
    return null;
  }, [events]);

  const partialEvents = useMemo(
    () => events.filter((event) => event.event === "partial"),
    [events]
  );
  const streamErrorEvents = useMemo(
    () => events.filter((event) => event.event === "error"),
    [events]
  );

  const latestPartialByModel = useMemo(() => {
    const result = {};
    partialEvents.forEach((event) => {
      if (event.model) {
        result[event.model] = event;
      }
    });
    return result;
  }, [partialEvents]);

  const summaryErrors = summary?.result?.errors || [];
  const errorItems = useMemo(() => {
    const requestItems = requestError
      ? [
          {
            source: "request",
            model: null,
            node: null,
            detail: requestError.detail,
            error_code: requestError.error_code || "REQUEST_ERROR"
          }
        ]
      : [];
    const streamItems = streamErrorEvents.map((event) => ({
      source: "stream",
      model: event.model || null,
      node: event.node || null,
      detail: event.detail || "Unknown stream error",
      error_code: event.error_code || "STREAM_ERROR"
    }));
    const summaryItems = summaryErrors.map((event) => ({
      source: "summary",
      model: event.model || null,
      node: event.node || null,
      detail: event.detail || "Unknown summary error",
      error_code: event.error_code || "SUMMARY_ERROR"
    }));
    return uniqBy([...requestItems, ...streamItems, ...summaryItems], (item) =>
      [item.source, item.model, item.node, item.error_code, item.detail].join("::")
    );
  }, [requestError, streamErrorEvents, summaryErrors]);

  const summarySuccessCount = summary?.result?.success_count ?? null;
  const summaryErrorCount = summary?.result?.error_count ?? null;
  const summaryErrorModels = summary?.result?.error_models || [];

  const answerCards = useMemo(() => {
    const summaryAnswers = summary?.result?.answers || {};
    const summaryStatuses = summary?.result?.api_status || {};
    const summaryDurations = summary?.result?.durations_ms || {};
    const summarySources = summary?.result?.sources || {};
    const orderedModels = summary?.result?.order || Object.keys(latestPartialByModel);

    return orderedModels
      .filter(Boolean)
      .map((model) => {
        const latestEvent = latestPartialByModel[model] || {};
        const answer = summaryAnswers[model] || latestEvent.answer || "";
        return {
          model,
          answer,
          status: summaryStatuses[model] || latestEvent.status || null,
          elapsedMs: summaryDurations[model] || latestEvent.elapsed_ms || null,
          source: summarySources[model] || latestEvent.source || null,
          responseMeta:
            summary?.result?.response_meta?.[model] || latestEvent.response_meta || null
        };
      })
      .filter((card) => card.answer || card.status || card.source || card.responseMeta);
  }, [latestPartialByModel, summary]);

  const successfulModelNames = useMemo(() => {
    const summaryModels = summary?.result?.success_models;
    if (Array.isArray(summaryModels) && summaryModels.length > 0) {
      return summaryModels;
    }
    return answerCards.filter((card) => isSuccessStatus(card.status)).map((card) => card.model);
  }, [answerCards, summary]);

  function toggleProvider(provider) {
    setSelectedProviders((current) =>
      current.includes(provider)
        ? current.filter((item) => item !== provider)
        : [...current, provider]
    );
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setEvents([]);
    setRequestError(null);
    setIsLoading(true);

    try {
      const response = await fetch(`${apiBaseUrl}/api/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question,
          active_providers: selectedProviders
        })
      });

      if (!response.ok) {
        let payload = null;
        try {
          payload = await response.json();
        } catch {
          payload = await response.text();
        }
        setRequestError({
          detail: toErrorMessage(payload),
          error_code: payload?.error_code || `HTTP_${response.status}`
        });
        return;
      }

      if (!response.body) {
        setRequestError({
          detail: "The backend response did not include a readable stream.",
          error_code: "NO_RESPONSE_BODY"
        });
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        buffer = parseNdjsonChunk(buffer, (streamEvent) => {
          setEvents((current) => [...current, streamEvent]);
        });
      }

      if (buffer.trim()) {
        setEvents((current) => [...current, JSON.parse(buffer)]);
      }
    } catch (submitError) {
      setRequestError({
        detail: submitError instanceof Error ? submitError.message : String(submitError),
        error_code: "NETWORK_ERROR"
      });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">Local-First Compare-AI</p>
        <h1>Readable multi-model answers, not raw stream noise.</h1>
        <p className="lede">
          Ask one question, stream model outputs from FastAPI, and review answers in structured
          cards with explicit error tracking.
        </p>
        <div className="hero-strip">
          <div className="hero-pill">
            <span className="pill-label">Mode</span>
            <strong>Chat compare</strong>
          </div>
          <div className="hero-pill">
            <span className="pill-label">Target API</span>
            <strong>{apiBaseUrl}</strong>
          </div>
          <div className="hero-pill">
            <span className="pill-label">Selection</span>
            <strong>{selectedProviders.length} providers</strong>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <form onSubmit={handleSubmit} className="stack">
            <div className="mode-copy">
              <h2>Compare live model answers</h2>
              <p>
                The app now exposes only the chat comparison path. Prompt evaluation has been
                removed from the active runtime.
              </p>
            </div>
            <label className="field">
              <span>FastAPI Base URL</span>
              <input value={apiBaseUrl} onChange={(event) => setApiBaseUrl(event.target.value)} />
            </label>
            <label className="field">
              <span>Question</span>
              <textarea
                rows="7"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask one question and compare how the selected providers answer it."
              />
            </label>
            <div className="field">
              <span>Providers</span>
              <div className="provider-grid">
                {providerOptions.map((provider) => (
                  <label key={provider} className="provider-pill">
                    <input
                      type="checkbox"
                      checked={selectedProviders.includes(provider)}
                      onChange={() => toggleProvider(provider)}
                    />
                    <span className="provider-text">
                      <strong>{providerLabels[provider]}</strong>
                      <small>{providerDescriptions[provider]}</small>
                    </span>
                  </label>
                ))}
              </div>
            </div>
            <button className="primary" type="submit" disabled={isLoading || !question.trim()}>
              {isLoading ? "Streaming..." : "Run compare"}
            </button>
          </form>
        </section>

        <section className="panel stack">
          <div className="section-header">
            <h2>Responses</h2>
            <p>{events.length} event(s)</p>
          </div>

          <div className="stat-row">
            <div className="stat-card">
              <span>LLM success</span>
              <strong>{summarySuccessCount ?? successfulModelNames.length}</strong>
            </div>
            <div className="stat-card">
              <span>Answers shown</span>
              <strong>{answerCards.length}</strong>
            </div>
            <div className="stat-card">
              <span>Stream events</span>
              <strong>{partialEvents.length}</strong>
            </div>
            <div className="stat-card error-card">
              <span>Errors</span>
              <strong>{summaryErrorCount ?? errorItems.length}</strong>
            </div>
          </div>

          {summary ? (
            <section className="summary-banner">
              <div className="summary-banner-card">
                <span className="pill-label">Backend summary</span>
                <strong>
                  {summarySuccessCount ?? successfulModelNames.length} succeeded,{" "}
                  {summaryErrorCount ?? errorItems.length} failed
                </strong>
              </div>
              <div className="summary-banner-card">
                <span className="pill-label">Error models</span>
                <strong>
                  {summaryErrorModels.length > 0 ? summaryErrorModels.join(", ") : "None"}
                </strong>
              </div>
            </section>
          ) : null}

          {errorItems.length > 0 ? (
            <section className="stack">
              <div className="section-header">
                <h3>Errors</h3>
                <p>Request, stream, and summary failures</p>
              </div>
              <div className="error-list">
                {errorItems.map((item, index) => (
                  <article key={`${item.source}-${item.error_code}-${index}`} className="error-panel">
                    <div className="event-meta">
                      <strong>{item.error_code}</strong>
                      <span>{item.model || item.node || item.source}</span>
                    </div>
                    <p>{item.detail}</p>
                  </article>
                ))}
              </div>
            </section>
          ) : null}

          {answerCards.length === 0 ? (
            <div className="empty-state">
              <h3>No answers yet</h3>
              <p>Submit a question to fill this panel with per-model answer cards and stream data.</p>
            </div>
          ) : (
            <>
              <div className="success-strip">
                <span className="pill-label">Successful models</span>
                <div className="success-badges">
                  {successfulModelNames.length > 0 ? (
                    successfulModelNames.map((model) => (
                      <span key={model} className="success-badge">
                        {model}
                      </span>
                    ))
                  ) : (
                    <span className="muted-copy">No successful model responses yet.</span>
                  )}
                </div>
              </div>
              <div className="response-list">
                {answerCards.map((card) => (
                  <article key={card.model} className="response-row">
                    <div className="response-head">
                      <span className="response-time">
                        {card.elapsedMs ? `${card.elapsedMs} ms` : "streaming"}
                      </span>
                      <span className="response-sep">|</span>
                      <strong className="response-company">{card.model}</strong>
                      <span className="response-sep">|</span>
                      <span className={`status-chip ${statusLabel(card.status)}`}>
                        {statusLabel(card.status)}
                      </span>
                    </div>
                    <div className="response-body">
                      <span className="response-label">Responses:</span>
                      <div className="response-text">{card.answer || "No answer content yet."}</div>
                    </div>
                    {card.source ? (
                      <div className="response-source">Source: {card.source}</div>
                    ) : null}
                    {card.responseMeta ? (
                      <details className="details-box">
                        <summary>Response metadata</summary>
                        <pre>{JSON.stringify(card.responseMeta, null, 2)}</pre>
                      </details>
                    ) : null}
                  </article>
                ))}
              </div>
            </>
          )}

          <details className="details-box">
            <summary>Raw stream events</summary>
            <div className="event-list">
              {events.map((streamEvent, index) => (
                <article key={`${streamEvent.event}-${index}`} className="event-card">
                  <div className="event-meta">
                    <strong>{streamEvent.event}</strong>
                    <span>{streamEvent.model || streamEvent.node || "system"}</span>
                  </div>
                  <pre>{JSON.stringify(streamEvent, null, 2)}</pre>
                </article>
              ))}
            </div>
          </details>
        </section>
      </main>
    </div>
  );
}
