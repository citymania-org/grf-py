NOTE: Live reloading is a highly experimental feature that may not work that well. Especially it's known to not update changes to python files.

# Prerequisites
1. Install `watchdog` module: `pip instal watchdog==3.0.0`
2. Use CityMania patchpack 14.0-RC1 or later: https://citymania.org/downloads

# Setup
1. Change `network.admin_password` in `secrets.cfg` to some non-empty value:
```ini
[network]
admin_password = pw
```
2. Check the value of `network.server_admin_port` in `openttd.cfg`, it can be any port, 3977 by default.
3. Use `grf.main` function in your code to generate grf instead of `NewGRF.write`:
```python
grf.main(g, 'super.grf')
```
4. Using CityMania patchpack start the game in *multiplayer* mode with necessary settings. Keep the server private so random players don't join it.
5. Use `python generate.py watch --live-reload=pw` to build grf and start watching for changes. Full format for `--live-reload` is `pw@server:port`, by default it's using server 'localhost' and port 3977.
