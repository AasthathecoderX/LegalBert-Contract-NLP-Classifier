document.addEventListener('DOMContentLoaded', () => {

  /* =========================
     DOM REFERENCES
  ========================= */
  const pdfInput = document.getElementById('pdfInput');
  const chooseFileBtn = document.getElementById('chooseFileBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const clearBtn = document.getElementById('clearBtn');
  const fileInfo = document.getElementById('fileInfo');
  const outputContainer = document.getElementById('outputContainer');
  const outputBox = document.getElementById('outputBox');
  const resultsSummary = document.getElementById('resultsSummary');
  const loadingOverlay = document.getElementById('loadingOverlay');

  console.log('✅ analyze.js loaded');

  if (!pdfInput || !chooseFileBtn || !analyzeBtn) {
    console.error('❌ Required DOM elements missing');
    return;
  }

  /* =========================
     STATE
  ========================= */
  let selectedFile = null;
  let allClauses = [];
  let showingAll = false;

  /* =========================
     VIEW MORE BUTTON
  ========================= */
  const viewMoreWrapper = document.createElement('div');
  viewMoreWrapper.className = 'view-more-wrapper';

  const viewMoreBtn = document.createElement('button');
  viewMoreBtn.className = 'btn-secondary';
  viewMoreBtn.textContent = 'View more clauses';

  viewMoreWrapper.appendChild(viewMoreBtn);

  viewMoreBtn.addEventListener('click', () => {
    showingAll = !showingAll;
    renderClauseList(showingAll);
  });

  /* =========================
     FILE HANDLING
  ========================= */
  chooseFileBtn.addEventListener('click', () => {
    pdfInput.click();
  });

  pdfInput.addEventListener('change', () => {
    if (pdfInput.files.length > 0) {
      selectedFile = pdfInput.files[0];
      fileInfo.textContent = `Selected: ${selectedFile.name}`;
      analyzeBtn.disabled = false;
    }
  });

  clearBtn.addEventListener('click', () => {
    selectedFile = null;
    pdfInput.value = '';
    fileInfo.textContent = '';
    analyzeBtn.disabled = true;
    outputContainer.style.display = 'none';
    outputBox.innerHTML = '<p class="placeholder">Results will appear here...</p>';
    resultsSummary.innerHTML = '';
    allClauses = [];
    showingAll = false;
  });

  /* =========================
     ANALYZE DOCUMENT
  ========================= */
  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    loadingOverlay.style.display = 'flex';

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      const result = await res.json();

      loadingOverlay.style.display = 'none';

      if (!result.success) {
        alert('Analysis failed');
        return;
      }

      outputContainer.style.display = 'block';
      renderResults(result.data);

    } catch (err) {
      loadingOverlay.style.display = 'none';
      console.error(err);
      alert('Server error during analysis');
    }
  });

  /* =========================
     RENDER RESULTS
  ========================= */
  function renderResults(data) {
    const summary = data.contract_analysis.summary_statistics;

    resultsSummary.innerHTML = `
      <div>Total clauses: ${summary.total_clauses}</div>
      <div>Clauses with labels: ${summary.clauses_with_labels}</div>
      <div>Total entities: ${summary.total_entities}</div>
      <div>High importance clauses: ${summary.high_importance_clauses}</div>
    `;

    allClauses = (data.contract_analysis.all_clauses || [])
      .filter(c => c.predicted_labels?.length);

    showingAll = false;
    renderClauseList(false);
  }

  /* =========================
     RENDER CLAUSE LIST
  ========================= */
  function renderClauseList(showAll) {
    outputBox.innerHTML = '';

    if (!allClauses.length) {
      outputBox.innerHTML = '<p class="placeholder">No important clauses found.</p>';
      return;
    }

    const clauses = showAll ? allClauses : allClauses.slice(0, 10);

    clauses.forEach((c, index) => {
      const clauseNum = typeof c.clause_id === 'number'
        ? c.clause_id + 1
        : index + 1;

      const labelObj = c.predicted_labels[0];
      const label = labelObj.label.replace(/-Answer$/i, '');
      const score = labelObj.score.toFixed(2);

      const clauseDiv = document.createElement('div');
      clauseDiv.className = 'entity-item legal';

      clauseDiv.innerHTML = `
        <div class="entity-label legal">
          Clause #${clauseNum} (Score: ${c.importance_score.toFixed(2)})
        </div>
        <div class="entity-text">
          ${escapeHtml(c.text).slice(0, 150)}...
        </div>
        <div style="margin-top:8px;font-weight:600;color:var(--primary-color);">
          Clause type: ${label} (${score})
        </div>
      `;

      if (c.entities?.length) {
        const entTitle = document.createElement('div');
        entTitle.innerHTML = '<strong>Entities:</strong>';
        entTitle.style.marginTop = '10px';
        clauseDiv.appendChild(entTitle);

        c.entities.slice(0, 10).forEach(ent => {
          const entDiv = document.createElement('div');
          const entLabel = (ent.label || ent.type || 'unknown').toLowerCase();

          entDiv.className = `entity-item ${entLabel}`;
          entDiv.innerHTML = `
            <span class="entity-label ${entLabel}">
              ${entLabel.toUpperCase()}
            </span>
            <span class="entity-text">
              ${escapeHtml(ent.text || ent.entity)}
            </span>
          `;
          clauseDiv.appendChild(entDiv);
        });
      }

      outputBox.appendChild(clauseDiv);
    });

    viewMoreBtn.textContent = showAll
      ? 'View top 10 clauses'
      : 'View more clauses';

    outputBox.appendChild(viewMoreWrapper);
  }

  /* =========================
     UTILS
  ========================= */
  function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/[&<>"']/g, m =>
      ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;' }[m])
    );
  }

});
