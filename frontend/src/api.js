const BASE = '/api';

async function request(path, options = {}) {
  const res = await fetch(BASE + path, options);
  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error(`Server error (HTTP ${res.status})`);
  }
  if (!res.ok) throw new Error(data?.detail || 'Request failed');
  return data;
}

export const getStatus = () => request('/status');

// Documents belong to a session — upload adds them to that session's index.
export const uploadDocuments = (sessionId, files) => {
  const form = new FormData();
  files.forEach(f => form.append('files', f));
  return request(`/sessions/${sessionId}/index`, { method: 'POST', body: form });
};

export const getSessions = () => request('/sessions');

export const getSession = (id) => request(`/sessions/${id}`);

export const createSession = (name) =>
  request('/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });

export const renameSession = (id, name) =>
  request(`/sessions/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });

export const deleteSession = (id) =>
  request(`/sessions/${id}`, { method: 'DELETE' });

export const getMessages = (id) => request(`/sessions/${id}/messages`);

export const getGraph = (sessionId) => request(`/sessions/${sessionId}/graph`);

export const querySession = (id, question) =>
  request(`/sessions/${id}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
