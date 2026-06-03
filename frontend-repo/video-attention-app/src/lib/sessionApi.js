async function requestJson(url, options) {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers ?? {}),
    },
    ...options,
  })

  const payload = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(payload.message || 'Request failed')
  }

  return payload
}

export function startSession({ name, email }) {
  return requestJson('/api/users/sessions', {
    method: 'POST',
    body: JSON.stringify({ name, email }),
  })
}

export function recordPrompt({ sessionId, name, email, videoKey }) {
  return requestJson(`/api/users/sessions/${sessionId}/prompts`, {
    method: 'POST',
    body: JSON.stringify({ name, email, videoKey }),
  })
}

export function completeSession({ sessionId }) {
  return requestJson(`/api/users/sessions/${sessionId}/complete`, {
    method: 'POST',
  })
}