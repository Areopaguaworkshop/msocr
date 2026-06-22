document.addEventListener('DOMContentLoaded', async () => {
  const elements = {
    list: document.getElementById('session-list'),
    langSelect: document.getElementById('lang-select'),
    scriptVar: document.getElementById('script-variant'),
    fragPath: document.getElementById('fragment-path'),
    createForm: document.getElementById('create-form')
  };

  async function loadLanguages() {
    try {
      const res = await fetch('/api/languages');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const langs = await res.json();
      elements.langSelect.innerHTML = langs.map(l => `<option value="${l.code}">${l.code} (${l.direction})</option>`).join('');
    } catch (err) {
      console.error('Failed to load languages:', err);
    }
  }

  async function loadSessions() {
    let res;
    try {
      res = await fetch('/api/sessions');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      elements.list.innerHTML = `<div class="error-state">Failed to load sessions: ${err.message}</div>`;
      return;
    }
    // ponytail: JSON.parse on bad bytes throws an opaque "Unexpected token"
    // — surface the raw text so the cause (e.g. a proxy inserting a footer)
    // is visible in the console and the page.
    let sessions;
    try {
      sessions = await res.json();
    } catch (err) {
      const raw = await res.text().catch(() => '<unreadable>');
      console.error('Failed to parse /api/sessions JSON:', err, 'raw body:', raw.slice(0, 500));
      elements.list.innerHTML = `<div class="error-state">Failed to load sessions: ${err.message}</div>`;
      return;
    }

    if (!Array.isArray(sessions) || sessions.length === 0) {
      elements.list.innerHTML = '<div class="empty-state">No sessions yet — create one below.</div>';
      return;
    }

    elements.list.innerHTML = sessions.map(s => `
      <div class="session-card">
        <img src="/api/sessions/${s.session_id}/image" alt="Fragment">
        <div class="session-info">
          <span class="session-id">${s.session_id}</span>
          <span class="session-meta">${s.language} · ${s.script_variant}</span>
          <span class="session-meta">${s.source}</span>
          <span class="session-meta">${s.line_count} lines · Updated ${timeAgo(s.updated_at)}</span>
        </div>
        <div class="session-footer">
          <a class="button primary" href="/ui/${s.session_id}">Open</a>
        </div>
      </div>
    `).join('');
  }

  function timeAgo(dateStr) {
    const diff = Math.floor((new Date() - new Date(dateStr)) / 1000);
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return new Date(dateStr).toLocaleDateString();
  }

  elements.createForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
      language: elements.langSelect.value,
      script_variant: elements.scriptVar.value,
      ingestion_path: 'local_file',
      source: elements.fragPath.value,
      crop_manuscript_area: false
    };

    const res = await fetch('/api/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      window.location.href = `/ui/${data.session_id}`;
    } else {
      const err = await res.json();
      alert(`Error creating session: ${err.detail || 'Unknown error'}`);
    }
  });

  await Promise.allSettled([loadLanguages(), loadSessions()]);
});
