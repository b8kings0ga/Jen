# Jen

Jen is a local hold-to-talk voice assistant extracted from Gjallarhorn.

## Run

```sh
uv run scripts/jen_voice.py --preset step
```

The default gateway is still the local Gjallarhorn-compatible API:

```sh
--gjallarhorn-base-url http://localhost:4000/v1
```

## Controls

- Hold Right Option: planned quality mode.
- Hold Right Command: simple fast mode.
- Double-tap either key: text input.
- Release, then tap the same key quickly: cancel.

## Dashboard

The dashboard starts on:

```text
http://127.0.0.1:8765/
```

## Stop

If running under screen:

```sh
screen -S jen_voice -X quit
```

Fallback:

```sh
pkill -f 'jen_voice.py'
pkill -f 'voice_gjallarhorn_say.py'
```

## Compatibility

`scripts/voice_gjallarhorn_say.py` remains as a compatibility wrapper for old commands.
