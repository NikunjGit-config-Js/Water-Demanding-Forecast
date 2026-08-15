# Permissions

## Automatically Allowed

Agents may:
- read project files
- create/edit project source files
- create tests
- run Python
- run pytest
- install ordinary project dependencies
- perform EDA
- train models
- run Optuna experiments
- use GPU
- create plots
- save artifacts
- inspect git status
- inspect git diff
- create local commits/checkpoints when explicitly enabled by supervisor
- retry failed code
- fix implementation errors
- write reports

## Must Ask User First

Agents must request permission before:
- deleting original datasets
- deleting large portions of the repository
- force pushing
- pushing publicly
- exposing credentials
- requesting/storing API keys
- using paid cloud services
- purchasing resources
- public deployment
- modifying GitHub history
- starting external Selenium/web scraping
- changing the fundamental approved experiment methodology
- destructive system operations

## Secrets

Never commit:
- API keys
- passwords
- tokens
- credentials

Use environment variables or ignored local configuration.

