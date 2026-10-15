"""Download the Silverbox record SNLS80mV.mat into this folder.

Official source: https://www.nonlinearbenchmark.org/ (SilverboxFiles.zip).
If the official download is not reachable, a byte-identical copy that is
committed in a public GitHub repository is used as a fallback.  The MD5 of
the file used for every number in the report is checked at the end.
"""
import hashlib, io, os, sys, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "SNLS80mV.mat")
MD5 = "f23cbdb4edf41e3107f60154082690db"

OFFICIAL_ZIP = "https://www.nonlinearbenchmark.org/FILES/BENCHMARKS/SILVERBOX/SilverboxFiles.zip"
MIRROR = "https://raw.githubusercontent.com/liouvill/PgDMM/main/silverbox_pgdmm/SNLS80mV.mat"


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def try_official():
    raw = urllib.request.urlopen(OFFICIAL_ZIP, timeout=30).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = [n for n in z.namelist() if n.endswith("SNLS80mV.mat")][0]
        with open(TARGET, "wb") as f:
            f.write(z.read(name))


def try_mirror():
    urllib.request.urlretrieve(MIRROR, TARGET)


if __name__ == "__main__":
    if not os.path.exists(TARGET):
        for fn in (try_official, try_mirror):
            try:
                fn()
                print("downloaded via", fn.__name__)
                break
            except Exception as e:  # noqa: BLE001
                print(fn.__name__, "failed:", e)
    if not os.path.exists(TARGET):
        sys.exit("could not download SNLS80mV.mat")
    h = md5(TARGET)
    print("md5", h, "OK" if h == MD5 else "(differs from the file used in the report)")
