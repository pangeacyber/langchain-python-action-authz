# User-based Access Control with Agent Actions for LangChain in Python

An example Python app demonstrating how to integrate Pangea's [AuthZ][] service
into a LangChain app to apply user-based authorization to control access to
actions executed via agents.

## Prerequisites

- Python v3.12 or greater.
- pip v24.2 or [uv][] v0.4.24.
- A [Pangea account][Pangea signup] with AuthZ enabled.
- An [OpenAI API key][OpenAI API keys].

## Setup

### Pangea AuthZ

The setup in AuthZ should look just like that of [a previous example][langchain-python-file-authz],
with the addition of a new resource type for the DuckDuckGo action:

| Name        | Permissions |
| ----------- | ----------- |
| engineering | read        |
| finance     | read        |
| duckduckgo  | read        |

and grant "read" permission to, for example, the finance role:

| Resource type | Permissions (read) |
| ------------- | ------------------ |
| engineering   | ❌                 |
| finance       | ✔️                 |
| duckduckgo    | ✔️                 |

### Repository

```shell
git clone https://github.com/pangeacyber/langchain-python-action-authz.git
cd langchain-python-action-authz
```

If using pip:

```shell
python -m venv .venv
source .venv/bin/activate
pip install .
```

Or, if using uv:

```shell
uv sync
source .venv/bin/activate
```

The sample can then be executed with:

```shell
python -m langchain_action_authz
```

## Usage

```
Usage: python -m langchain_action_authz [OPTIONS] PROMPT

Options:
  --user TEXT              Unique username to simulate retrieval as.
                           [required]
  --authz-token SECRET     Pangea AuthZ API token. May also be set via the
                           `PANGEA_AUTHZ_TOKEN` environment variable.
                           [required]
  --pangea-domain TEXT     Pangea API domain. May also be set via the
                           `PANGEA_DOMAIN` environment variable.  [default:
                           aws.us.pangea.cloud; required]
  --model TEXT             OpenAI model.  [default: gpt-4o-mini; required]
  --openai-api-key SECRET  OpenAI API key. May also be set via the
                           `OPENAI_API_KEY` environment variable.  [required]
  --help                   Show this message and exit.
```

For this example, we have set it up such that user Alice, in Engineering, does
not have permission to make DuckDuckGo searches:

```
python -m langchain_action_authz --user alice "What is MFA?"
```

```
User alice is not authorized to use this tool.
```

But Bob, in Finance, can:

```
python -m langchain_action_authz --user bob "What is MFA?"
```

```
MFA stands for Multi-Factor Authentication, a security mechanism that requires users to provide multiple forms of verification to access an account or system, enhancing security beyond just a password.
```

[AuthZ]: https://pangea.cloud/docs/authz/
[Pangea signup]: https://pangea.cloud/signup
[OpenAI API keys]: https://platform.openai.com/api-keys
[uv]: https://docs.astral.sh/uv/
[langchain-python-file-authz]: https://github.com/pangeacyber/langchain-python-file-authz
