import { useEffect, useState, type FormEvent } from 'react'

type Mode = { label: string; path: string; hint: string }

const MODES: Mode[] = [
  { label: 'Agent', path: '/chat/decide', hint: 'Routes to direct answer, docs, web search, or both' },
  { label: 'Ask docs', path: '/chat/ask', hint: 'Answers only from uploaded documents' },
  { label: 'Chat', path: '/chat/', hint: 'Plain LLM chat' },
]

type Message = {
  role: 'user' | 'assistant' | 'error'
  text: string
  sources?: string[]
  mode?: string
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

async function post(path: string, body: BodyInit, headers?: HeadersInit) {
  const res = await fetch(`/api${path}`, { method: 'POST', body, headers })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : `Request failed (${res.status})`
    throw new Error(detail)
  }
  return data
}

export default function App() {
  const [mode, setMode] = useState(MODES[0])
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('')
  const [threadId, setThreadId] = useState(loadThreadId)

  // Restore the conversation after a page refresh.
  useEffect(() => {
    fetch(`/api/chat/${threadId}/history`)
      .then((res) => (res.ok ? res.json() : { messages: [] }))
      .then((data: { messages: { role: 'user' | 'assistant'; text: string }[] }) =>
        setMessages(data.messages.map((m) => ({ role: m.role, text: m.text }))),
      )
      .catch(() => {})
  }, [threadId])

  function newChat() {
    setThreadId(newThreadId())
    setMessages([])
  }

  async function send(e: FormEvent) {
    e.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setInput('')
    setMessages((m) => [...m, { role: 'user', text, mode: mode.label }])
    setLoading(true)
    try {
      const data = await post(mode.path, JSON.stringify({ message: text, thread_id: threadId }), {
        'Content-Type': 'application/json',
      })
      setMessages((m) => [...m, { role: 'assistant', text: data.response, sources: data.sources }])
    } catch (err) {
      setMessages((m) => [...m, { role: 'error', text: (err as Error).message }])
    } finally {
      setLoading(false)
    }
  }

  async function upload(file: File | undefined) {
    if (!file) return
    setUploadStatus(`Uploading ${file.name}…`)
    const form = new FormData()
    form.append('file', file)
    try {
      const data = await post('/docs', form)
      setUploadStatus(`Indexed ${data.filename} (${data.chunk_count} chunks)`)
    } catch (err) {
      setUploadStatus(`Upload failed: ${(err as Error).message}`)
    }
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
              accept=".md,.txt,.html"
              onChange={(e) => {
                upload(e.target.files?.[0])
                e.target.value = ''
              }}
            />
          </label>
        </div>
      </header>
      {uploadStatus && <p className="status" role="status">{uploadStatus}</p>}

      <nav className="modes" role="tablist">
        {MODES.map((m) => (
          <button
            key={m.path}
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
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.mode && <span className="tag">{m.mode}</span>}
            <p>{m.text}</p>
            {m.sources && m.sources.length > 0 && (
              <p className="sources">Sources: {m.sources.join(', ')}</p>
            )}
          </div>
        ))}
        {loading && <div className="msg assistant thinking">Thinking…</div>}
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
