"""Regenerate the CC BY 4.0 digits asset, never train during a site build."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from forge_ml.datasets import stratified_split

root = Path(__file__).resolve().parents[1]
x, y = load_digits(return_X_y=True)
pixels, labels = x.astype(np.uint8), y.astype('<i4')
obj = {'version':1, 'rows':len(y), 'shape':[1797,64], 'pixels':pixels.ravel().tolist(),
       'labels':labels.tolist(), 'splits':stratified_split(labels,2026), 'splitSeed':2026,
       'checksum':hashlib.sha256(pixels.tobytes()+labels.tobytes()).hexdigest(),
       'preprocessing':'float32(pixel) / 16; fixed range, no fitted statistics',
       'provenance':{'creators':'E. Alpaydin and C. Kaynak (1998)', 'source':'sklearn.datasets.load_digits, scikit-learn 1.7.2; UCI optical recognition test subset',
                     'url':'https://doi.org/10.24432/C50P49','license':'CC BY 4.0',
                     'licenseUrl':'https://creativecommons.org/licenses/by/4.0/',
                     'modifications':'Serialized original pixels; added fixed stratified 70/15/15 split indices.'}}
out = root/'web/public/assets/digits.json'
out.write_text(json.dumps(obj,separators=(',',':')))
print(json.dumps({'file':str(out),'bytes':out.stat().st_size,'checksum':obj['checksum'],
                  'splits':{k:len(v) for k,v in obj['splits'].items()}},indent=2))
