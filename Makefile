# Thin shim over the justfile. `make <target>` delegates to `just <target>`.
# Most targets accept positional args via `just`; from make, set vars:
#   make ingest YEARS=1
#   make agent EPISODES=3 HOURS=24 SPEED=80 SEED=11

YEARS    ?= 2
EPISODES ?= 1
HOURS    ?= 24
SPEED    ?= 20
SEED     ?= 42

.PHONY: default redis redis-stop ingest api agent ui test typecheck dev up down logs

default:
	@just --list

redis:
	@just redis

redis-stop:
	@just redis-stop

ingest:
	@just ingest $(YEARS)

api:
	@just api

agent:
	@just agent $(EPISODES) $(HOURS) $(SPEED) $(SEED)

ui:
	@just ui

test:
	@just test

typecheck:
	@just typecheck

dev:
	@just dev

up:
	@just up

down:
	@just down

logs:
	@just logs
