"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
const learnerUserId = localStorage.getItem("langbuddy.userId") || "local-user";
localStorage.setItem("langbuddy.userId", learnerUserId);

const state = {
  bookId: localStorage.getItem("langbuddy.bookId") || localStorage.getItem("bookId") || "",
  books: [],
  activeWord: "",
  detailWord: "",
  groups: [],
  reviewWords: [],
  quizSessionId: "",
  currentView: "assistant",
};

const viewLoaders = {
  assistant: () => Promise.all([loadChatHistory(), loadOverview()]),
  vocabulary: () => Promise.all([loadBooks(), loadGroups()]),
  study: () => loadOverview(),
  quiz: async () => {},
  materials: () => loadMaterials(),
};

async function api(url, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-User-ID", learnerUserId);
  const response = await fetch(url, { ...options, headers });
  const text = await response.text();
  let data = {};
  if (text) {
    try { data = JSON.parse(text); }
    catch { data = { detail: text }; }
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function showToast(message, kind = "success") {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast show${kind === "error" ? " error" : ""}`;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.className = "toast"; }, 3400);
}

function readableError(error, fallback) {
  console.error(error);
  showToast(error?.message || fallback, "error");
}

function requireBook() {
  if (state.bookId) return true;
  showToast("Select a vocabulary book first.", "error");
  switchView("vocabulary");
  return false;
}

function setBusy(element, busy, label = "Working…") {
  if (!element) return;
  if (busy) {
    element.dataset.originalLabel = element.textContent;
    element.textContent = label;
    element.disabled = true;
  } else {
    element.textContent = element.dataset.originalLabel || element.textContent;
    element.disabled = false;
  }
}

function bookLabel(book) {
  return book?.source || book?.bookId || "Untitled book";
}

function updateBookContext() {
  const active = state.books.find((book) => book.bookId === state.bookId);
  $("#bookContext").textContent = active ? bookLabel(active) : "No book selected";
  $("#activeBookName").textContent = active ? bookLabel(active) : "No book selected";
  $("#activeBookMeta").textContent = active
    ? `${active.bookId} · ${Number(active.group_count || 0)} morphology groups`
    : "Choose or import a vocabulary book.";
  $("#bookSelect").value = state.bookId;
  $$(".book-row").forEach((row) => row.classList.toggle("active", row.dataset.bookId === state.bookId));
}

function switchView(viewName) {
  if (!viewLoaders[viewName]) return;
  state.currentView = viewName;
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${viewName}`));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === viewName));
  const view = $(`#view-${viewName}`);
  $("#viewTitle").textContent = view.dataset.title;
  $("#viewEyebrow").textContent = view.dataset.eyebrow;
  document.body.classList.remove("nav-open");
  viewLoaders[viewName]().catch((error) => readableError(error, "Could not refresh this section."));
}

async function selectBook(bookId, { refresh = true } = {}) {
  state.bookId = bookId || "";
  state.activeWord = "";
  resetWordDetail();
  localStorage.setItem("langbuddy.bookId", state.bookId);
  localStorage.setItem("bookId", state.bookId);
  updateFocusUI();
  updateBookContext();
  if (!refresh || !state.bookId) return;
  await Promise.all([loadChatHistory(), loadOverview(), loadMaterials()]);
  if (state.currentView === "vocabulary") await loadGroups();
}

async function loadBooks() {
  const data = await api("/api/books");
  state.books = data.items || [];
  if (state.bookId && !state.books.some((book) => book.bookId === state.bookId)) state.bookId = "";
  if (!state.bookId && state.books.length) state.bookId = state.books[0].bookId;

  const select = $("#bookSelect");
  select.innerHTML = '<option value="">Select a book</option>' + state.books.map((book) =>
    `<option value="${escapeHtml(book.bookId)}">${escapeHtml(bookLabel(book))}</option>`
  ).join("");

  const list = $("#bookList");
  if (!state.books.length) {
    list.innerHTML = '<div class="empty-state small"><p>No books yet. Import a CSV to begin.</p></div>';
  } else {
    list.innerHTML = state.books.map((book) => `
      <div class="book-row${book.bookId === state.bookId ? " active" : ""}" data-book-id="${escapeHtml(book.bookId)}">
        <div><strong>${escapeHtml(bookLabel(book))}</strong><div class="row-meta">${escapeHtml(book.bookId)} · ${Number(book.group_count || 0)} groups</div></div>
        <button class="button secondary compact use-book" type="button">${book.bookId === state.bookId ? "Selected" : "Use book"}</button>
      </div>`).join("");
    $$(".use-book", list).forEach((button) => button.addEventListener("click", () => {
      const bookId = button.closest(".book-row").dataset.bookId;
      selectBook(bookId).then(() => {
        renderBookList();
        setLibraryManager(false);
        showToast("Active book updated.");
      });
    }));
  }
  updateBookContext();
  return state.books;
}

function renderBookList() {
  const list = $("#bookList");
  if (!list || !state.books.length) return;
  $$(".book-row", list).forEach((row) => {
    const active = row.dataset.bookId === state.bookId;
    row.classList.toggle("active", active);
    const button = $(".use-book", row);
    if (button) button.textContent = active ? "Selected" : "Use book";
  });
}

function setLibraryManager(open) {
  const manager = $("#libraryManager");
  const button = $("#toggleLibrary");
  manager.classList.toggle("hidden", !open);
  button.setAttribute("aria-expanded", String(open));
  button.textContent = open ? "Close library" : "Manage library";
}

function resetWordDetail() {
  state.detailWord = "";
  $("#wordSearchInput").value = "";
  $("#wordDetailTitle").textContent = "Choose a word";
  $("#wordDetailHint").textContent = "Select a word in the explorer to see its structure, examples, learning notes, and source.";
  $("#wordDetailActions").classList.add("hidden");
  $("#explanationResult").innerHTML = '<div class="empty-state small"><p>Word details will appear here.</p></div>';
}

async function importBook(event) {
  event.preventDefault();
  const file = $("#bookFile").files[0];
  if (!file) return showToast("Choose a CSV file first.", "error");
  const button = $("#bookUploadForm button[type='submit']");
  setBusy(button, true, "Importing…");
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("lang", $("#bookLanguage").value);
    const data = await api("/api/books/import", { method: "POST", body: form });
    $("#bookUploadForm").reset();
    await loadBooks();
    await selectBook(data.bookId);
    renderBookList();
    showToast("Vocabulary book imported.");
  } catch (error) { readableError(error, "Book import failed."); }
  finally { setBusy(button, false); }
}

function setActiveWord(word) {
  state.activeWord = String(word || "").trim();
  $("#focusWord").value = state.activeWord;
  updateFocusUI();
}

function updateFocusUI() {
  const active = Boolean(state.activeWord);
  $("#clearFocus").classList.toggle("hidden", !active);
  $("#focusNote").textContent = active
    ? `“${state.activeWord}” is extra context — you can still ask about anything else.`
    : "No active word — the assistant can still use learner tools, memory, and materials.";
}

function applyFocus() {
  setActiveWord($("#focusWord").value);
  showToast(state.activeWord ? `Focus set to “${state.activeWord}”.` : "Word focus cleared.");
}

function renderMessages(messages) {
  const box = $("#chatMessages");
  if (!messages?.length) {
    box.innerHTML = '<div class="empty-state compact-empty"><span class="empty-icon">✦</span><h3>Start anywhere</h3><p>Try “What should I review today?” or ask about any language concept.</p></div>';
    return;
  }
  box.innerHTML = "";
  for (const message of messages) {
    const row = document.createElement("div");
    row.className = `message ${message.role === "user" ? "user" : "assistant"}`;
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.textContent = message.role === "user" ? "You" : "L";
    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.content || "";
    row.append(avatar, content);
    box.appendChild(row);
  }
  box.scrollTop = box.scrollHeight;
}

function appendPendingMessage() {
  const box = $("#chatMessages");
  $(".empty-state", box)?.remove();
  const row = document.createElement("div");
  row.className = "message assistant pending";
  row.id = "pendingMessage";
  row.innerHTML = '<span class="message-avatar">L</span><div class="message-content">Thinking<span class="typing-dots"></span></div>';
  box.appendChild(row);
  box.scrollTop = box.scrollHeight;
}

async function loadChatHistory() {
  if (!state.bookId) return renderMessages([]);
  const data = await api(`/api/books/${encodeURIComponent(state.bookId)}/chat/history`);
  renderMessages(data.messages || []);
}

async function sendChat(event) {
  event?.preventDefault();
  if (!requireBook()) return;
  const input = $("#chatInput");
  const text = input.value.trim();
  if (!text) return;
  const button = $("#sendMessage");
  button.disabled = true;
  input.disabled = true;
  appendPendingMessage();
  try {
    const data = await api(`/api/books/${encodeURIComponent(state.bookId)}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, active_word: state.activeWord || null }),
    });
    input.value = "";
    input.style.height = "auto";
    renderMessages(data.messages || []);
    await loadOverview();
  } catch (error) {
    $("#pendingMessage")?.remove();
    readableError(error, "The assistant could not respond.");
  } finally {
    button.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

async function resetConversation() {
  if (!requireBook()) return;
  if (!window.confirm("Start a new global conversation? Word-specific chat history will stay unchanged.")) return;
  try {
    const data = await api(`/api/books/${encodeURIComponent(state.bookId)}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force_new: true, active_word: state.activeWord || null }),
    });
    renderMessages(data.messages || []);
    showToast("New conversation started.");
  } catch (error) { readableError(error, "Conversation could not be reset."); }
}

function usePrompt(prompt) {
  switchView("assistant");
  $("#chatInput").value = prompt;
  $("#chatInput").focus();
}

async function loadOverview() {
  if (!state.bookId) {
    $("#overviewDue").textContent = "—";
    $("#overviewWeak").textContent = "—";
    $("#overviewMaterials").textContent = "—";
    $("#overviewGoal").textContent = "Select a book to see learner data.";
    $("#overviewWeakWords").innerHTML = '<span class="subtle">No active learner context.</span>';
    $("#navDueCount").textContent = "0";
    return;
  }
  const data = await api(`/api/books/${encodeURIComponent(state.bookId)}/overview`);
  const dueCount = Number(data.due?.total_matches || 0);
  const weakCount = Number(data.weak?.total_matches || 0);
  $("#overviewDue").textContent = dueCount;
  $("#overviewWeak").textContent = weakCount;
  $("#overviewMaterials").textContent = Number(data.material_count || 0);
  $("#navDueCount").textContent = dueCount;
  const goals = data.memory?.goals || [];
  $("#overviewGoal").textContent = goals[0] || "No saved goal yet. Tell the assistant, “My goal is …”";
  const weak = data.weak?.items || [];
  const tags = $("#overviewWeakWords");
  tags.innerHTML = weak.length
    ? weak.map((item) => `<button class="tag" type="button" data-word="${escapeHtml(item.word)}">${escapeHtml(item.word)}</button>`).join("")
    : '<span class="subtle">No weak-word evidence yet.</span>';
  $$(".tag[data-word]", tags).forEach((button) => button.addEventListener("click", () => {
    setActiveWord(button.dataset.word); switchView("assistant");
  }));
}

async function loadGroups() {
  const list = $("#groupList");
  if (!state.bookId) {
    state.groups = [];
    list.innerHTML = '<div class="empty-state small"><p>Select a book to explore vocabulary groups.</p></div>';
    return;
  }
  const query = $("#groupSearch").value.trim();
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  const suffix = params.toString() ? `?${params}` : "";
  const data = await api(`/api/books/${encodeURIComponent(state.bookId)}/groups${suffix}`);
  state.groups = data.groups || [];
  renderGroups();
}

function renderGroups() {
  const list = $("#groupList");
  if (!state.groups.length) {
    list.innerHTML = '<div class="empty-state small"><p>No matching groups.</p></div>';
    return;
  }
  list.innerHTML = state.groups.map((group) => `
    <button class="group-button" type="button" data-label="${escapeHtml(group.label)}">
      <span>${escapeHtml(group.label)}</span><small>${Number(group.count || 0)}</small>
    </button>`).join("");
  $$(".group-button", list).forEach((button) => button.addEventListener("click", () => loadGroupWords(button.dataset.label, button)));
}

async function loadGroupWords(label, button) {
  try {
    $$(".group-button").forEach((item) => item.classList.toggle("active", item === button));
    const data = await api(`/api/groups/${encodeURIComponent(state.bookId)}/${encodeURIComponent(label)}`);
    renderWords(data.words || [], label);
  } catch (error) { readableError(error, "Group could not be loaded."); }
}

function renderWords(words, label = "Words") {
  const box = $("#wordList");
  if (!words.length) {
    box.innerHTML = '<div class="empty-state small"><p>No words in this group.</p></div>';
    return;
  }
  box.innerHTML = words.map((item) => {
    const word = typeof item === "string" ? item : item.word;
    const decomposition = typeof item === "object" ? item.decomposition || "" : "";
    return `<button class="word-chip" type="button" data-word="${escapeHtml(word)}" title="${escapeHtml(decomposition || label)}">${escapeHtml(word)}</button>`;
  }).join("");
  $$(".word-chip", box).forEach((button) => button.addEventListener("click", () => explainWord(button.dataset.word)));
}

async function loadUngrouped() {
  if (!requireBook()) return;
  try {
    const data = await api(`/api/books/${encodeURIComponent(state.bookId)}/ungrouped`);
    renderWords(data.items || [], "Ungrouped");
    $$(".group-button").forEach((item) => item.classList.remove("active"));
  } catch (error) { readableError(error, "Ungrouped words could not be loaded."); }
}

function renderExplanation(word, explain, source) {
  const displayValue = (value) => {
    if (value == null || value === "") return "—";
    if (typeof value === "object") {
      return value.text || value.en || value.hook || value.mnemonic || value.label || JSON.stringify(value);
    }
    return String(value);
  };
  const list = (value) => Array.isArray(value) && value.length
    ? `<ul>${value.map((item) => `<li>${escapeHtml(displayValue(typeof item === "object" ? item.en || item : item))}</li>`).join("")}</ul>`
    : "<p>—</p>";
  const examples = explain.examples || [];
  $("#explanationResult").innerHTML = `
    <div class="explain-grid">
      <div class="explain-item"><h4>Word</h4><p><strong>${escapeHtml(word)}</strong></p></div>
      <div class="explain-item"><h4>Structure</h4><p>${escapeHtml(displayValue(explain.decomposition || explain.root || explain.affix))}</p></div>
      <div class="explain-item wide"><h4>Memory hook</h4><p>${escapeHtml(displayValue(explain.hook || explain.mnemonic))}</p></div>
      <div class="explain-item"><h4>Collocations</h4>${list(explain.collocations)}</div>
      <div class="explain-item"><h4>Pitfalls / confusables</h4>${list(explain.pitfalls || explain.confusables)}</div>
      <div class="explain-item wide"><h4>Examples</h4>${list(examples)}</div>
      <div class="explain-item wide"><h4>Source</h4><p>${escapeHtml(source || "deterministic morphology")}</p></div>
    </div>`;
}

async function explainWord(word) {
  if (!requireBook()) return;
  const cleanWord = String(word || $("#wordSearchInput").value || "").trim();
  if (!cleanWord) return showToast("Enter a word to explain.", "error");
  state.detailWord = cleanWord;
  $("#wordSearchInput").value = cleanWord;
  $("#wordDetailTitle").textContent = cleanWord;
  $("#wordDetailHint").textContent = "Structure, usage guidance, examples, and learner actions for this word.";
  $("#wordDetailActions").classList.remove("hidden");
  $("#explanationResult").innerHTML = '<div class="empty-state small"><p>Building an explanation…</p></div>';
  try {
    const data = await api(`/api/vocab/${encodeURIComponent(state.bookId)}/${encodeURIComponent(cleanWord)}/explain?ai=1`);
    renderExplanation(cleanWord, data.explain || {}, data.source || (data.cached ? "saved explanation" : "AI + morphology"));
    $("#explanationResult").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) { readableError(error, "Explanation could not be generated."); }
}

async function generateWordFeedback() {
  if (!requireBook()) return;
  const word = state.detailWord;
  if (!word) return showToast("Choose a word before generating feedback.", "error");
  const button = $("#generateFeedback");
  setBusy(button, true, "Generating…");
  try {
    const data = await api(`/api/vocab/${encodeURIComponent(state.bookId)}/${encodeURIComponent(word)}/feedback`, { method: "POST" });
    const issues = (data.issues || []).map((item) => typeof item === "string" ? item : item.detail || item.code || JSON.stringify(item));
    $("#explanationResult").innerHTML = `
      <div class="explain-grid">
        <div class="explain-item"><h4>Word</h4><p><strong>${escapeHtml(word)}</strong></p></div>
        <div class="explain-item"><h4>Mastery</h4><p>${Math.round(Number(data.mastery || 0) * 100)}%</p></div>
        <div class="explain-item"><h4>Strengths</h4>${renderSimpleList(data.strengths)}</div>
        <div class="explain-item"><h4>Issues</h4>${renderSimpleList(issues)}</div>
        <div class="explain-item wide"><h4>Tips</h4>${renderSimpleList(data.tips)}</div>
      </div>`;
  } catch (error) { readableError(error, "Feedback could not be generated."); }
  finally { setBusy(button, false); }
}

function renderSimpleList(items) {
  if (!Array.isArray(items) || !items.length) return "<p>—</p>";
  return `<ul>${items.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("")}</ul>`;
}

async function loadReview() {
  if (!requireBook()) return;
  const button = $("#loadReview");
  setBusy(button, true, "Loading…");
  try {
    const data = await api(`/api/review/today/${encodeURIComponent(state.bookId)}`);
    const due = data.due || [];
    const fresh = data.new || [];
    state.reviewWords = [...due, ...fresh].map((item) => item.word).filter(Boolean);
    $("#dueListCount").textContent = String(due.length);
    $("#newListCount").textContent = String(fresh.length);
    $("#reviewSummary").innerHTML = `
      <span class="summary-pill">${due.length} due</span>
      <span class="summary-pill">${fresh.length} new</span>
      <span class="summary-pill">Daily limit ${data.stats?.new_quota ?? 10}</span>`;
    renderReviewCards($("#dueReviewList"), due, "No cards are due right now.");
    renderReviewCards($("#newReviewList"), fresh, "No new cards available.");
    await loadOverview();
  } catch (error) { readableError(error, "Today’s review could not be loaded."); }
  finally { setBusy(button, false); }
}

function renderReviewCards(container, items, emptyMessage) {
  if (!items.length) {
    container.innerHTML = `<div class="empty-state small"><p>${escapeHtml(emptyMessage)}</p></div>`;
    return;
  }
  container.innerHTML = items.map((item) => `
    <article class="review-card" data-word="${escapeHtml(item.word)}">
      <div class="review-card-head"><strong title="Use as chat focus">${escapeHtml(item.word)}</strong><button class="button text compact explain-review" type="button">Explain</button></div>
      <div class="grade-buttons" aria-label="Rate ${escapeHtml(item.word)} from 0 to 5">
        ${[0,1,2,3,4,5].map((grade) => `<button type="button" data-grade="${grade}" title="Grade ${grade}">${grade}</button>`).join("")}
      </div>
      <div class="review-status">Choose 0 (forgot) to 5 (easy).</div>
    </article>`).join("");
  $$(".review-card", container).forEach((card) => {
    const word = card.dataset.word;
    $("strong", card).addEventListener("click", () => { setActiveWord(word); switchView("assistant"); });
    $(".explain-review", card).addEventListener("click", () => { switchView("vocabulary"); explainWord(word); });
    $$("[data-grade]", card).forEach((button) => button.addEventListener("click", () => saveReviewGrade(card, word, Number(button.dataset.grade))));
  });
}

async function saveReviewGrade(card, word, grade) {
  const buttons = $$("[data-grade]", card);
  buttons.forEach((button) => { button.disabled = true; });
  $(".review-status", card).textContent = "Saving…";
  try {
    const data = await api(`/api/review/${encodeURIComponent(state.bookId)}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ word, grade }),
    });
    buttons.forEach((button) => { button.style.background = Number(button.dataset.grade) === grade ? "#e7f3ed" : ""; });
    $(".review-status", card).textContent = `Saved · next review ${new Date(data.due_at).toLocaleDateString()}`;
    await loadOverview();
  } catch (error) {
    $(".review-status", card).textContent = "Could not save this grade.";
    readableError(error, "Review grade could not be saved.");
  } finally { buttons.forEach((button) => { button.disabled = false; }); }
}

async function startQuiz(event) {
  event.preventDefault();
  if (!requireBook()) return;
  const button = $("#quizSetup button[type='submit']");
  setBusy(button, true, "Building quiz…");
  let words = $("#quizWords").value.split(",").map((word) => word.trim()).filter(Boolean);
  if (!words.length && state.activeWord) words = [state.activeWord];
  if (!words.length) words = state.reviewWords.slice(0, 10);
  try {
    const data = await api("/api/session/quiz/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ words, total: Number($("#quizTotal").value) }),
    });
    state.quizSessionId = data.session_id || "";
    renderQuiz(data.items || data.paper || []);
  } catch (error) { readableError(error, "Quiz could not be started."); }
  finally { setBusy(button, false); }
}

function renderQuiz(items) {
  const area = $("#quizArea");
  if (!items.length) {
    area.innerHTML = '<div class="empty-state"><p>No quiz items were generated.</p></div>';
    return;
  }
  area.innerHTML = `<form id="quizPaper" class="quiz-paper">
    ${items.map((question, index) => `
      <article class="quiz-question" data-id="${escapeHtml(question.id)}">
        <h3>${index + 1}. ${escapeHtml(question.stem || "Choose the best answer.")}</h3>
        ${(question.options || []).length ? question.options.map((option, optionIndex) => `
          <label class="option"><input type="radio" name="question-${index}" value="${optionIndex}"><span>${escapeHtml(option)}</span></label>`).join("")
          : `<input type="text" name="question-${index}" placeholder="Type your answer">`}
      </article>`).join("")}
    <div class="quiz-actions"><button class="button primary" type="submit">Submit answers</button></div>
  </form>`;
  $("#quizPaper").addEventListener("submit", submitQuiz);
}

async function submitQuiz(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const button = $("button[type='submit']", form);
  setBusy(button, true, "Checking…");
  const answers = $$(".quiz-question", form).map((question) => {
    const checked = $("input[type='radio']:checked", question);
    const text = $("input[type='text']", question);
    return { id: question.dataset.id, answer: checked ? Number(checked.value) : (text?.value || "").trim() };
  });
  try {
    const data = await api("/api/session/quiz/submit", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ book_id: state.bookId, session_id: state.quizSessionId, answers }),
    });
    renderQuizFeedback(data.feedback || {});
    await loadOverview();
  } catch (error) { readableError(error, "Quiz could not be submitted."); setBusy(button, false); }
}

function renderQuizFeedback(feedback) {
  const mastery = Math.round(Number(feedback.mastery || 0) * 100);
  const explanations = feedback.explanations || [];
  $("#quizArea").innerHTML = `<div class="feedback-summary">
    <div class="score-card"><strong>${mastery}%</strong><span>${Number(feedback.correct || 0)} / ${Number(feedback.total || 0)} correct</span></div>
    <div><h2>Quiz feedback</h2><p class="subtle">${escapeHtml(feedback.summary_line || "Your learner state has been updated.")}</p>
      <div class="feedback-details">${explanations.slice(0, 8).map((item) => `<div class="feedback-row"><strong>${escapeHtml(item.id || "Question")}</strong> · ${item.ok ? "Correct" : "Review"}<br>${escapeHtml(item.why || "")}</div>`).join("")}</div>
    </div>
  </div>`;
}

function clearQuiz() {
  state.quizSessionId = "";
  $("#quizArea").innerHTML = '<div class="empty-state"><span class="empty-icon">✓</span><h3>Ready when you are</h3><p>Choose target words, or leave the field empty to use your current focus and review queue.</p></div>';
}

async function loadMaterials() {
  const list = $("#materialList");
  if (!state.bookId) {
    list.innerHTML = '<div class="empty-state small"><p>Select a book to manage its learning materials.</p></div>';
    return;
  }
  const data = await api(`/api/materials/${encodeURIComponent(state.bookId)}`);
  const items = data.items || [];
  if (!items.length) {
    list.innerHTML = '<div class="empty-state"><span class="empty-icon">▤</span><h3>No materials yet</h3><p>Upload notes or a PDF, then ask the assistant to use the source.</p></div>';
    return;
  }
  list.innerHTML = items.map((item) => {
    const extension = String(item.source_name || "file").split(".").pop();
    return `<div class="material-row" data-document-id="${escapeHtml(item.document_id)}">
      <div class="material-row-main"><span class="file-kind">${escapeHtml(extension)}</span><div><strong>${escapeHtml(item.source_name)}</strong><div class="row-meta">${Number(item.chunk_count || 0)} chunks · ${Number(item.char_count || 0).toLocaleString()} characters</div></div></div>
      <button class="button danger-text compact delete-material" type="button">Delete</button>
    </div>`;
  }).join("");
  $$(".delete-material", list).forEach((button) => button.addEventListener("click", () => deleteMaterial(button.closest(".material-row").dataset.documentId)));
}

async function uploadMaterial(event) {
  event.preventDefault();
  if (!requireBook()) return;
  const file = $("#materialFile").files[0];
  if (!file) return showToast("Choose a material file first.", "error");
  const button = $("#materialForm button[type='submit']");
  setBusy(button, true, "Uploading…");
  try {
    const form = new FormData(); form.append("file", file);
    await api(`/api/materials/${encodeURIComponent(state.bookId)}`, { method: "POST", body: form });
    $("#materialForm").reset();
    await Promise.all([loadMaterials(), loadOverview()]);
    showToast("Material uploaded and ready for retrieval.");
  } catch (error) { readableError(error, "Material upload failed."); }
  finally { setBusy(button, false); }
}

async function deleteMaterial(documentId) {
  if (!window.confirm("Delete this material from the selected book?")) return;
  try {
    await api(`/api/materials/${encodeURIComponent(state.bookId)}/${encodeURIComponent(documentId)}`, { method: "DELETE" });
    await Promise.all([loadMaterials(), loadOverview()]);
    showToast("Material deleted.");
  } catch (error) { readableError(error, "Material could not be deleted."); }
}

async function clearMaterials() {
  if (!requireBook()) return;
  if (!window.confirm("Delete all uploaded materials for this book?")) return;
  try {
    await api(`/api/materials/${encodeURIComponent(state.bookId)}`, { method: "DELETE" });
    await Promise.all([loadMaterials(), loadOverview()]);
    showToast("All materials for this book were cleared.");
  } catch (error) { readableError(error, "Materials could not be cleared."); }
}

function bindEvents() {
  $$(".nav-item").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $$("[data-go]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.go)));
  $$("[data-prompt]").forEach((button) => button.addEventListener("click", () => usePrompt(button.dataset.prompt)));
  $("#bookSelect").addEventListener("change", (event) => selectBook(event.target.value));
  $("#bookUploadForm").addEventListener("submit", importBook);
  $("#chatForm").addEventListener("submit", sendChat);
  $("#chatInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendChat(event); }
  });
  $("#chatInput").addEventListener("input", (event) => {
    event.target.style.height = "auto";
    event.target.style.height = `${Math.min(event.target.scrollHeight, 150)}px`;
  });
  $("#applyFocus").addEventListener("click", applyFocus);
  $("#focusWord").addEventListener("keydown", (event) => { if (event.key === "Enter") applyFocus(); });
  $("#clearFocus").addEventListener("click", () => { setActiveWord(""); showToast("Word focus cleared."); });
  $("#newConversation").addEventListener("click", resetConversation);
  $("#loadGroups").addEventListener("click", () => loadGroups().catch((error) => readableError(error, "Groups could not be loaded.")));
  $("#groupSearch").addEventListener("input", () => {
    clearTimeout(bindEvents.groupTimer);
    bindEvents.groupTimer = setTimeout(() => loadGroups().catch(() => {}), 250);
  });
  $("#showUngrouped").addEventListener("click", loadUngrouped);
  $("#toggleLibrary").addEventListener("click", () => setLibraryManager($("#libraryManager").classList.contains("hidden")));
  $("#wordSearchForm").addEventListener("submit", (event) => { event.preventDefault(); explainWord($("#wordSearchInput").value); });
  $("#focusExplainedWord").addEventListener("click", () => {
    const word = state.detailWord;
    if (!word) return showToast("Choose a word first.", "error");
    setActiveWord(word); switchView("assistant"); showToast(`Focus set to “${word}”.`);
  });
  $("#generateFeedback").addEventListener("click", generateWordFeedback);
  $("#loadReview").addEventListener("click", loadReview);
  $("#quizSetup").addEventListener("submit", startQuiz);
  $("#clearQuiz").addEventListener("click", clearQuiz);
  $("#materialForm").addEventListener("submit", uploadMaterial);
  $("#clearMaterials").addEventListener("click", clearMaterials);
  $("#refreshView").addEventListener("click", () => viewLoaders[state.currentView]().catch((error) => readableError(error, "Refresh failed.")));
  $("#mobileMenu").addEventListener("click", () => document.body.classList.add("nav-open"));
  $("#sidebarScrim").addEventListener("click", () => document.body.classList.remove("nav-open"));
}

async function init() {
  bindEvents();
  updateFocusUI();
  try {
    await loadBooks();
    if (state.bookId) await selectBook(state.bookId);
    else await loadOverview();
  } catch (error) { readableError(error, "LangBuddy could not load the workspace."); }
}

document.addEventListener("DOMContentLoaded", init);
