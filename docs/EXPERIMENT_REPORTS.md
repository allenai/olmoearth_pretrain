# Daily Experiment Reports

Automated daily reports of your Beaker experiments, sent to Slack.

## Admin Setup (One-Time)

These secrets are **shared by everyone** and only need to be set up once by a repo admin:

| Secret | Purpose | How to get |
|--------|---------|------------|
| `CURSOR_API_KEY` | Runs the AI agent | [Cursor dashboard](https://cursor.com) |
| `SLACK_WEBHOOK_URL` | Posts reports to Slack | [Slack App settings](https://api.slack.com/apps) → Incoming Webhooks |

Go to [Repository Settings → Secrets → Actions](../../settings/secrets/actions) to add these.

---

## Per-User Setup

Each user needs their **own Beaker token** (this is what identifies whose experiments to report on).

### 1. Get your Beaker token

```bash
beaker account token
```

Copy the output.

### 2. Add your token as a GitHub secret

1. Go to [Repository Settings → Secrets → Actions](../../settings/secrets/actions)
2. Click "New repository secret"
3. Name: `BEAKER_TOKEN_<YOUR_USERNAME>` (uppercase)
   - Example: If your username is `joer`, use `BEAKER_TOKEN_JOER`
4. Value: Paste your beaker token
5. Click "Add secret"

### 3. Add yourself to the user list

Edit `.github/experiment-report-users.json` and add your username:

```json
{
  "users": [
    "henryh",
    "joer",
    "your_username"
  ]
}
```

Commit and push the change.

### 4. You're done!

You'll receive a Slack report every weekday at 5am PST with:
- Running experiments and their progress
- Failed experiments with error analysis
- Actionable TODO items

## Manual Trigger

To run a report immediately:

1. Go to [Actions → Daily Experiment Report](../../actions/workflows/daily-experiment-report.yaml)
2. Click "Run workflow"
3. Optionally override the user list (comma-separated): `henryh,joer`
4. Click "Run workflow"

## Troubleshooting

### "No BEAKER_TOKEN_XXX secret found"

Make sure your secret name matches your username exactly (uppercase):
- Username: `joer` → Secret: `BEAKER_TOKEN_JOER`
- Username: `henryh` → Secret: `BEAKER_TOKEN_HENRYH`

### Not receiving reports

1. Check you're in `.github/experiment-report-users.json`
2. Check the [workflow runs](../../actions/workflows/daily-experiment-report.yaml) for errors
3. Verify your beaker token is still valid: `beaker account whoami`

### Token expired

Regenerate with `beaker account token` and update the secret.
