from pathlib import Path
import hashlib
import json
import sys
import pickle


class SpriteCache:
    HASH_FORMAT = f'{{:0{sys.hash_info.width // 4}x}}'
    HASH_MASK = (1 << sys.hash_info.width) - 1

    def __init__(self, path):
        self.path = Path(path)
        self.index_path = self.path / 'index.json'
        self._old_keys = set()
        self._index = {}
        self._new_keys = set()

    def load(self, clean_build):
        self.path.mkdir(parents=True, exist_ok=True)
        if self.index_path.exists():
            try:
                self._index = json.load(open(self.index_path))
            except Exception as e:
                print(f'WARNING: Error loading cache index: {e}')
                print('WARNING: Unable to remove unused cache files automatically, delete .cache directory if that is an issue')
                self._index = {}
            self._old_keys = set(self._index.keys())
        else:
            self._index = {}
            self._old_keys = set()
        self._new_keys = set()
        if clean_build:
            # Clean build, don't use index (but we still load it for old keys).
            self._index = {}

    def save(self):
        for k in self._old_keys:
            if k in self._new_keys:
                continue
            try:
                path = self.path / k
                path.unlink()
            except Exception as e:
                print(f'WARNING(internal): Broken sprite cache entry {k} (delete fail): {e}')

            if k in self._index:
                del self._index[k]
        json.dump(self._index, open(self.index_path, 'w'), indent=4)
        self._old_keys = self._new_keys
        self._new_keys = set()

    @staticmethod
    def hexdigest(hash_data):
        s = json.dumps(
            hash_data,
            ensure_ascii=False,
            sort_keys=True,
            indent=None,
            separators=(',', ':'),
        ).encode('utf-8')
        return hashlib.md5(s).hexdigest()[:16]

    def is_cached(self, hash_key):
        assert isinstance(hash_key, str) and len(hash_key) == 16, hash_key
        return hash_key in self._index

    def get(self, hash_key):
        assert isinstance(hash_key, str) and len(hash_key) == 16, hash_key
        if hash_key not in self._index:
            return None

        self._new_keys.add(hash_key)
        try:
            return open(self.path / hash_key, 'rb').read()
        except Exception as e:
            print(f'WARNING(internal): Broken sprite cache entry {hash_key} (get fail): {e}')

    def set(self, hash_key, data):
        assert isinstance(hash_key, str) and len(hash_key) == 16, hash_key
        self._new_keys.add(hash_key)
        with open(self.path / str(hash_key), 'wb') as f:
            f.write(data)
        self._index[hash_key] = True
