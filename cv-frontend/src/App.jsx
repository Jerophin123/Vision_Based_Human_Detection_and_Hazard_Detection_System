import { useEffect, useMemo, useRef, useState } from "react";

const API = import.meta.env.VITE_API || "http://127.0.0.1:8000";

export default function App() {
  // ---- logical slots (from --cams keys) ----
  const [camIds, setCamIds] = useState([]);
  const [devices, setDevices] = useState([]);
  const [binding, setBinding] = useState({});
  const [cams, setCams] = useState([]);
  const [selected, setSelected] = useState("");

  const [stats, setStats] = useState({ count: 0, last: "-" });
  const [events, setEvents] = useState([]);
  const [loadingCams, setLoadingCams] = useState(true);
  const [error, setError] = useState("");

  const [view, setView] = useState("grid"); // 'single' | 'grid'
  const [activating, setActivating] = useState(false);
  const [tab, setTab] = useState("video"); // 'video' | 'alerts'
  const audioRef = useRef(null);

  // --------- data loaders ----------
  const loadCamIds = () => {
    fetch(`${API}/cam_ids`)
      .then((r) => r.json())
      .then((d) => {
        const ids = d?.cam_ids || [];
        setCamIds(ids);
        setBinding((prev) => {
          const copy = { ...prev };
          ids.forEach((id) => {
            if (!(id in copy)) copy[id] = "";
          });
          return copy;
        });
      })
      .catch(() => setCamIds([]));
  };

  const loadDevices = () => {
    fetch(`${API}/devices`)
      .then((r) => r.json())
      .then((d) => setDevices(d?.devices || []))
      .catch(() => setDevices([]));
  };

  const loadActiveCams = () => {
    setLoadingCams(true);
    setError("");
    fetch(`${API}/cams`)
      .then((r) => r.json())
      .then((d) => {
        const list = d?.cams || [];
        setCams(list);
        if (list.length) {
          setSelected((prev) => (prev && list.includes(prev) ? prev : list[0]));
        } else {
          setSelected("");
        }
      })
      .catch((e) => setError(`Failed to load cameras: ${e.message}`))
      .finally(() => setLoadingCams(false));
  };

  const activate = () => {
    const map = {};
    camIds.forEach((id) => {
      if (binding[id]) map[id] = binding[id];
    });
    setActivating(true);
    fetch(`${API}/activate_map`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ map }),
    })
      .then((r) => r.json())
      .then(() => loadActiveCams())
      .finally(() => setActivating(false));
  };

  // --------- effects ----------
  useEffect(loadCamIds, []);
  useEffect(loadDevices, []);
  useEffect(loadActiveCams, []);

  useEffect(() => {
    const es = new EventSource(`${API}/events`);
    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      setEvents((prev) => [data, ...prev].slice(0, 300));
      if (audioRef.current) {
        audioRef.current.currentTime = 0;
        audioRef.current.play().catch(() => {});
      }
      fetch(`${API}/stats`).then((r) => r.json()).then(setStats);
    };
    return () => es.close();
  }, []);

  useEffect(() => {
    fetch(`${API}/stats`).then((r) => r.json()).then(setStats);
  }, []);

  const videoSrc = useMemo(
    () => (selected ? `${API}/video?cam=${encodeURIComponent(selected)}` : ""),
    [selected]
  );

  return (
    <div className="rzm-root">
      {/* theme & component styles (scoped) */}
      <style>{`
        :root {
          --bg: #090a0f;
          --panel: #0d0f15;
          --card: #0b0d12;
          --border: #1b2030;
          --text: #e7e9ee;
          --text-dim: #9aa3b2;
          --brand: #60a5fa;
          --brand-2: #34d399;
          --ok: #22c55e;
          --warn: #ef4444;
          --amber: #f59e0b;
          --shadow: 0 10px 30px rgba(0,0,0,.35);
        }
        .rzm-root {
          min-height: 100vh;
          color: var(--text);
          background:
            radial-gradient(60% 50% at 20% 0%, rgba(96,165,250,.12) 0%, rgba(0,0,0,0) 70%),
            radial-gradient(50% 40% at 85% 10%, rgba(52,211,153,.10) 0%, rgba(0,0,0,0) 70%),
            linear-gradient(180deg, #08090e 0%, #0a0b10 60%, #08090e 100%);
          display: flex; align-items: center; flex-direction: column;
        }
        .container { width: 100%; max-width: 1400px; }

        /* Header */
        .hdr {
          position: sticky; top: 0; z-index: 50; width: 100%;
          backdrop-filter: blur(8px);
          background: linear-gradient(180deg, rgba(10,11,16,.9), rgba(10,11,16,.75));
          border-bottom: 1px solid var(--border);
        }
        .hdr-inner { max-width: 1400px; margin: 0 auto; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
        .brand {
          display: flex; align-items: center; gap: 10px;
        }
        .pulse { width: 10px; height: 10px; border-radius: 9999px; background: var(--warn); box-shadow: 0 0 0 0 rgba(239,68,68,.8); animation: pulse 2s infinite; }
        @keyframes pulse { 0%{ box-shadow: 0 0 0 0 rgba(239,68,68,.5);} 70%{ box-shadow: 0 0 0 12px rgba(239,68,68,0);} 100%{ box-shadow: 0 0 0 0 rgba(239,68,68,0);} }

        /* Controls */
        .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 16px; box-shadow: var(--shadow); }
        .controls { display: grid; grid-template-columns: repeat(auto-fit,minmax(440px,1fr)); gap: 12px; align-items: center; }
        .row { display: flex; align-items: center; gap: 12px; }
        .slot { width: 120px; font-weight: 700; letter-spacing: .2px; color: var(--text); }

        /* Buttons & Inputs */
        .btn { appearance: none; background: #1b1f2a; color: #fff; border: 1px solid var(--border); padding: 10px 14px; border-radius: 12px; cursor: pointer; transition: transform .12s ease, box-shadow .12s ease, background .2s ease, border-color .2s ease; box-shadow: 0 4px 18px rgba(0,0,0,.25) inset; }
        .btn:hover { transform: translateY(-1px); border-color: #2a3349; }
        .btn:active { transform: translateY(0); }
        .btn.brand { background: linear-gradient(180deg, #3b82f6, #1d4ed8); border-color: #2a5bd3; }
        .btn.brand.active { box-shadow: 0 0 0 2px rgba(59,130,246,.25); }
        .btn.ok { background: linear-gradient(180deg, #22c55e, #16a34a); border-color: #177b3f; }
        .seg { display:inline-flex; background:#12141a; border:1px solid var(--border); border-radius:14px; padding:4px; }
        .seg button { background: transparent; border:none; color:#cfd6e6; padding:8px 14px; border-radius:10px; cursor:pointer; transition: background .2s; }
        .seg button[aria-pressed="true"]{ background:#1f2330; color:#fff; }

        select.sel { background:#151823; color:#fff; border:1px solid var(--border); border-radius: 12px; padding:10px 12px; min-width: 320px; outline: none; transition: border-color .2s, box-shadow .2s; }
        select.sel:focus { border-color:#3450ff; box-shadow: 0 0 0 3px rgba(52,80,255,.15); }

        /* Stats chip */
        .chip { font-size:12px; color: var(--text-dim); background: #11131a; padding: 6px 10px; border-radius: 999px; border:1px solid var(--border); }

        /* Main */
        .main { padding: 16px 24px; }

        /* Video cards */
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 16px; }
        .cam { border-radius: 16px; overflow: hidden; background:#000; border:1px solid var(--border); display:flex; flex-direction:column; box-shadow: var(--shadow); }
        .cam-h { padding: 10px 12px; background: linear-gradient(180deg, #0f121a, #0d1017); border-bottom: 1px solid var(--border); font-weight:700; text-align:center; letter-spacing:.2px; }
        .cam img { width:100%; height:360px; object-fit:cover; display:block; }
        .single { border-radius:16px; overflow:hidden; background:#000; border:1px solid var(--border); box-shadow: var(--shadow); }

        /* Alerts */
        .alerts { display:grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap:12px; }
        .alert { background: linear-gradient(180deg,#ffe2e2,#ffd6d6); color:#111; border-left: 6px solid var(--warn); border-radius: 14px; padding: 12px; box-shadow: 0 6px 18px rgba(239,68,68,.15); }
        .meta { font-size:12px; color:#333; }

        /* Empty states */
        .empty { color: var(--text-dim); text-align: center; padding: 24px; border: 1px dashed var(--border); border-radius: 14px; background: #0c0f15; }

        /* Shimmer for loaders */
        .shimmer { position: relative; overflow: hidden; background: #0f131c; border-radius: 14px; height: 220px; border:1px solid var(--border); }
        .shimmer::after { content:""; position:absolute; inset:0; background: linear-gradient(110deg, transparent 0%, rgba(255,255,255,.04) 40%, rgba(255,255,255,.08) 50%, rgba(255,255,255,.04) 60%, transparent 100%); animation: shimmer 1.4s linear infinite; }
        @keyframes shimmer { 100% { transform: translateX(100%); } }

        /* Footer */
        .ftr { opacity:.7; font-size:12px; padding: 18px 0 30px; text-align:center; color: var(--text-dim); }
      `}</style>

      <div className="container">
        {/* Header */}
        <header className="hdr">
          <div className="hdr-inner">
            <div className="brand">
              <span className="pulse" />
              <strong style={{ letterSpacing: 0.2 }}>Restricted Zone Monitor</strong>
            </div>

            {/* Tabs */}
            <div className="seg" role="tablist" aria-label="Primary tabs">
              <button
                role="tab"
                aria-selected={tab === "video"}
                aria-pressed={tab === "video"}
                onClick={() => setTab("video")}
                title="Live video and controls"
              >
                Video
              </button>
              <button
                role="tab"
                aria-selected={tab === "alerts"}
                aria-pressed={tab === "alerts"}
                onClick={() => setTab("alerts")}
                title="Recent intrusion alerts"
              >
                Alerts
              </button>
            </div>

            <div className="chip">
              Events: {stats.count} &nbsp;|&nbsp; Last: {stats.last}
            </div>
          </div>
        </header>

        {/* Content */}
        <main className="main">
          {tab === "video" && (
            <>
              {/* Controls card */}
              <section className="panel" aria-label="Camera controls">
                <div className="controls">
                  {camIds.length === 0 ? (
                    <div className="empty">⚙️ No camera slots available.</div>
                  ) : (
                    camIds.map((id) => (
                      <div key={id} className="row">
                        <div className="slot">{id}</div>
                        <select
                          className="sel"
                          value={binding[id] || ""}
                          onChange={(e) => setBinding((b) => ({ ...b, [id]: e.target.value }))}
                        >
                          <option value="">— choose device —</option>
                          {devices.map((d) => (
                            <option key={`${id}-${d.name}`} value={d.name}>
                              {d.name}
                            </option>
                          ))}
                        </select>
                      </div>
                    ))
                  )}
                </div>

                <div style={{ display: "flex", gap: 10, marginTop: 14, flexWrap: "wrap", justifyContent: "space-between" }}>
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                    <button
                      onClick={activate}
                      disabled={activating}
                      className="btn ok"
                      aria-busy={activating}
                    >
                      {activating ? "Activating…" : "Activate"}
                    </button>
                    <button onClick={loadDevices} className="btn" title="Reload device list">Refresh devices</button>
                    <button onClick={loadActiveCams} className="btn" title="Reload active cameras">Reload cameras</button>
                  </div>

                  {/* Single/grid & picker */}
                  <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                    <select
                      className="sel"
                      value={selected}
                      onChange={(e) => setSelected(e.target.value)}
                      disabled={!cams.length}
                      style={{ minWidth: 220, opacity: cams.length ? 1 : 0.6 }}
                      title="Current camera in single view"
                    >
                      {cams.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>

                    <div className="seg" role="group" aria-label="View mode">
                      <button aria-pressed={view === "single"} onClick={() => setView("single")}>Single</button>
                      <button aria-pressed={view === "grid"} onClick={() => setView("grid")}>Grid</button>
                    </div>
                  </div>
                </div>
              </section>

              {/* Errors / loading */}
              {error && (
                <div style={{ background: "#ffd6d6", color: "#111", borderLeft: "6px solid #dc2626", borderRadius: 12, padding: "10px 12px", margin: "12px 0", boxShadow: "0 6px 18px rgba(239,68,68,.15)" }}>
                  {error}
                </div>
              )}
              {loadingCams && (
                <div className="shimmer" aria-hidden="true" />
              )}

              {/* Video area */}
              {view === "single" ? (
                <div className="single">
                  {selected ? (
                    <img src={videoSrc} alt="live stream" />
                  ) : (
                    <div className="empty">🎥 No camera selected.</div>
                  )}
                </div>
              ) : (
                <div className="grid">
                  {cams.length === 0 && <div className="empty">🎥 No cameras configured.</div>}

                  {/* Side-by-side camera cards */}
                  <div style={{ display: "flex", gap: 16, width: "100%" }}>
                    {cams.map((c) => (
                      <div key={c} className="cam">
                        <div className="cam-h">{c}</div>
                        <img
                          src={`${API}/video?cam=${encodeURIComponent(c)}`}
                          alt={`${c} stream`}
                          loading="lazy"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {tab === "alerts" && (
            <section aria-label="Recent alerts">
              {events.length === 0 ? (
                <div className="empty">🔔 No alerts yet.</div>
              ) : (
                <div className="alerts">
                  {events.map((e, i) => (
                    <div key={i} className="alert">
                      <div>
                        <b>{e.label}</b> in <b>{e.zone_id}</b> from <b>{e.cam_id}</b>
                      </div>
                      <div className="meta">at {e.human_time} | bbox {JSON.stringify(e.bbox)}</div>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}
        </main>

        {/* Footer */}
        <div className="ftr">Press <kbd>TAB</kbd> to navigate controls • All times local</div>
      </div>

      {/* hidden audio for beeps */}
      <audio ref={audioRef} src={`${API}/beep`} preload="auto" />
    </div>
  );
}
