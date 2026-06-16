# Security & Compliance Notes

This document covers the security posture of DocuMind and the open items that
must be addressed before delivering it to a Chartered Accountancy / audit firm.

## ⚠️ Data residency — read before deploying with real client data

DocuMind sends document text and images to the **OpenAI API** (a third-party
service) for every tool: chat/RAG, OCR, the Financial Extractor, and the
classifier. The in-app consent banner (stored in `localStorage`) is **not**
sufficient for a CA firm's compliance obligations.

Before a CA firm uploads real client books/returns/working papers, resolve one
of the following:

- **Azure OpenAI** (or an equivalent enterprise tier) under a signed data
  processing agreement, with a no-training guarantee and a defined region. This
  is usually the fastest path to a compliant deployment.
- **Self-hosted / on-premise models** for the sensitive paths, so client data
  never leaves the firm's environment.

Relevant obligations to confirm with the firm's compliance owner:

- **ICAI confidentiality** requirements for client information.
- **Digital Personal Data Protection Act, 2023 (India)** for processing of
  personal data, including cross-border transfer.

Until this is resolved, treat the app as suitable only for **non-confidential or
synthetic documents** (demos, training, evaluation).

## Authentication

- All routes except `/login` and `/static` require session auth.
- **The initial admin will not be seeded with a weak password.** On a fresh
  database, `ADMIN_PASSWORD` must be set to a strong value (8+ chars, not
  `admin`) or the app refuses to create the admin account. For local
  development only, set `ALLOW_INSECURE_ADMIN=1` to bypass this check.
- Set `ADMIN_EMAIL` and `ADMIN_PASSWORD` in the deployment environment (e.g.
  Render env vars) before first boot.

### Recommended follow-ups (not yet implemented)

- **Rate limiting on `/login`** to slow credential-guessing.
- **Retention policy**: configurable auto-deletion of uploaded documents and
  chat history per client engagement.

## Data handling

- Uploaded documents are chunked and stored in PostgreSQL, scoped to the
  uploading user's session. Restarting the server does not clear stored data
  (it is persisted in the database).
- Chat answers cite the source **document and page number** so extracted facts
  can be verified against the original.
