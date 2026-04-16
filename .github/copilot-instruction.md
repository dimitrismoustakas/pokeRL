# Coding Instructions

## Core Principles

1. **Fail Loudly instead of failing silently** - Write code that tells us if something fails instead of having silent fallbacks just so the code runs, this will ruin the repo in the long run if it happens
2. **Mantain good structure** - Keep code related to each different thing in its own file, so we can keep track of what each piece of code does
3. **No hacky code** - Write code that solves the issue in principle, avoid 'hacky' code that bypasses issues but instead try to solve these issues
4. **Do not overdo it with argparse** - Adding argparse is fine but it should be limited because it otherwise encourages hacky code