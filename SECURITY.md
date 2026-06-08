# Security Policy

## Supported Versions

| Version | Supported          |
|---------|-------------------|
| 2.0.x   | ✅ Active support  |
| < 2.0   | ❌ Not supported   |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue,
please report it by emailing [INSERT EMAIL ADDRESS] — **do not** create a
public GitHub issue.

Please include the following details:

- Type of issue (e.g., remote code execution, SQL injection, XSS)
- Full paths of source files related to the issue
- Steps to reproduce the issue
- Proof of concept or exploit code (if available)
- Impact assessment

We will acknowledge receipt within 48 hours and provide an initial assessment
within 5 business days. We will keep you informed of the progress toward a fix.

## Best Practices for Users

1. **API Key**: When running in production, set `AITRAINER_API_KEY` to restrict
   API access.
2. **CORS**: Configure `AITRAINER_CORS_ORIGINS` to restrict which domains can
   access the API in production.
3. **Network**: By default the server binds to `127.0.0.1`. For production,
   use a reverse proxy (nginx, Caddy) and bind to the appropriate interface.
4. **Dependencies**: Keep dependencies updated by regularly running
   `pip install --upgrade -r backend/requirements.txt`.
