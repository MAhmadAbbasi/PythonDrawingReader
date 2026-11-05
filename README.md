# Drawing2Quote

Reads an electrical drawing (DXF, or a vector PDF exported from CAD) and suggests the home-automation solution
from the POS's **Inventory > Solution Packages** data. No AI / cloud service - only deterministic libraries
(ezdxf, PyMuPDF, shapely).

It is one half of a hybrid system:

```
POS page  Sales > Drawing to Solution
   │  upload, review counts, choose Standard / Premium, confirm "both" products, export Excel / PDF
   ▼
.NET backend  api/drawings/*            (login, company, file checks, the company's solution data)
   │  POST /extract (file)      POST /suggest {extraction, catalog, plan, choices, sizeId}
   ▼
Drawing2Quote  (this service, FastAPI on 127.0.0.1:8765)
```

## How a solution is worked out

1. **Read the drawing** (`app/extract`) - title block, floors, legend, rooms (closed polylines / drawn
   rectangles around room names), electrical points per room (DXF blocks, PDF symbol shapes matched to the
   legend), points outside rooms in "Unassigned", a confidence score per room.
2. **Building size** (`app/rules/building_size.py`) - the size written on the drawing ("10 MARLA"), else a
   small single-floor plan's bedrooms (BHK), else the largest floor's area as a plot size. Can be overridden.
3. **Package + drawing** (`app/rules/suggest.py`) - start from that size's package quantities, then replace the
   items `config/solution_rules.yaml` links to drawing points (switch boards from light/fan points, IR remotes
   from AC/TV points, curtain motors and rail, kitchen gas/smoke sensors ...). Rounded up.
4. **Plan** - Standard: Standard-only and "both" products, no questions. Premium: Premium-only products; each
   "both" product is confirmed first (use it, skip it, or use a Premium-only product serving the same points).

5. **No prices** - the result is products and quantities only. The POS's "Create Quotation" opens the normal
   quotation form with the linked inventory products and these quantities; prices, discounts and taxes are
   set there as for any quotation.

## Run

```
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8765
```

POS backend settings (`appsettings*.json`):

```json
"DrawingService": { "Url": "http://127.0.0.1:8765", "Key": "" }
```

Set the same key in the `D2Q_KEY` environment variable of this service if it is reachable from other machines.

## Command line

```
python -m app.cli plan.dxf                                     # what was read
python -m app.cli plan.pdf --overlay checked.pdf               # + a copy of the PDF with rooms/symbols marked
python -m app.cli plan.pdf --pages 6-8                         # only these pages
python -m app.cli plan.dxf --plan standard --catalog catalog.json
python -m app.cli plan.dxf --plan premium --choices all --db "Server=..;Database=..;User=..;Password=.." --tenant 3
```

`--db` reads the solution data straight from the POS database (MySQL / MariaDB).

## How a PDF is read (`app/extract/pdf_reader.py`, `pdf_geometry.py`)

* **Strokes** - every vector path is broken into single lines (curves as short lines), so it doesn't matter
  whether the CAD program put one symbol in many paths or thousands of lines in one path.
* **Legend** - each description row under LEGEND / SYMBOLS and the symbol drawn to its left: a shape, a shape
  with letters in it ("S", "TV"), or just letters ("AC").
* **Symbols** - small strokes that touch are grouped into one symbol. Each symbol is described by a shape
  signature that ignores position and rotation (size, ink length, how the ink spreads from the centre, line
  directions, curved/filled, letters inside) and matched to the closest legend symbol. A socket turned 90
  degrees on a wall still matches. Without a legend nothing is counted, but repeated shapes are reported.
* **Rooms** - traced from the walls: lines are tried from the heaviest line weight down (walls are drawn heavier
  than furniture, hatching, dimensions; dashed wiring is ignored), door openings up to 4 ft are closed by
  extending wall ends across the gap, and the enclosed space around each room name is the room. A space is
  accepted only with exactly one room name in it, away from the page edge (the outside), with a believable
  area. Else the smallest drawn rectangle around the name; else only the name's position.
* **Areas** - the size written in the room ("14'x16'"), else outline x scale ("SCALE 1:100"), else the scale
  worked out from the rooms that have a size written.
* Symbols drawn on a wall line count for the room they face. `--overlay` shows all of it on the drawing.

## Printed PDFs (no selectable text)

A PDF printed through a printer driver ("Adobe PDF", PScript5 / Distiller) has every word turned into lines.
Such pages are read with OCR (`app/extract/ocr.py`, RapidOCR on onnxruntime, offline; `pip install
rapidocr-onnxruntime`): the page is rendered at 200 dpi, word boxes are found tile by tile, merged across tiles,
and each word is read once. About 30 s per A3 sheet the first time; the result is cached on disk by the file's
content (`D2Q_OCR_CACHE`, default `%TEMP%\drawing2quote-ocr`), so the same file is read instantly afterwards.

Printer drivers also flatten the drawing: circles become many tiny lines, filled walls and dots become thin
slivers. The reader allows for that - wiring is separated from symbols (long straight lines and runs of pieces
joined end to end), legend tables (SYMBOLS | DESCRIPTION columns) are read cell by cell, stacked switch dots are
counted by their filled area, solid walls are used as areas, and names outside the building are ignored.

Several sheets of the same floor (lighting layout, power layout ...) are merged: the same room keeps the highest
count of each point type, so a point drawn on two sheets counts once.

Best results still come from DXF, or a PDF plotted with AutoCAD's "DWG To PDF.pc3" (text stays text).

## Scanned PDFs and pictures (JPG, PNG, TIFF) (`app/extract/raster_reader.py`)

A scan or a photo has only pixels. Each page is read from the image:

1. **Ink**: pixels darker than the paper in any colour channel, by more than the paper's own grain (uneven light
   in photos is evened out; faint lines are kept when they form a long stroke).
2. **Words**: OCR on a contrast-stretched copy (cached like printed PDFs). Words of 3+ characters are wiped from the ink.
3. **Tracing**: thick ink becomes filled shapes (walls cut into pieces without holes, touching switch dots split
   apart); thin ink is thinned to centre lines and traced into polylines that end where lines meet. Lines that
   meet are written as one path, so a wall network is long like in CAD. The result is a PDF page that
   `pdf_reader` reads for the title, floor, rooms and areas.
4. **Symbols**: each legend row's symbol is cut out of the picture and looked for over the building (chamfer
   matching on stroke outlines, at the four quarter turns). A quick pass at a small size finds the places; each
   place is then judged in full detail against the symbols that came close, so similar symbols (a 5A socket and
   a 15A socket) are told apart. A coloured legend symbol only matches ink of its colour; letters inside a
   matched symbol ("TV") are not taken as room names.

On the test plan printed and "scanned" at 300 dpi (tilted, blurred, grainy) about 95% of symbols come out right,
with no invented ones, and rooms are traced from the walls. A page takes about 1 minute (plus OCR the first time).
Scan straight, at 300 dpi, well lit: faint thin lines (e.g. pale colour hairlines from a printed CAD drawing)
read far worse than a crisp black print. `--overlay` draws the readings on the traced drawing.

## Adapting to your drawings

* `config/block_mapping.yaml` - block names / legend words -> point types. Add your architects' block names
  under `blocks:`; unmapped blocks are listed in the warnings after each upload.
* `config/solution_rules.yaml` - which package item each kind of point becomes, and how many.

## Tests

```
python -m pytest -q
```

`tests/fixtures.py` builds a DXF and a vector PDF (2 floors, 5 rooms, legend, a stray point) whose counts must be
read exactly; `tests/cad_fixture.py` builds a CAD-style PDF (double-line walls with door openings, furniture,
hatching, wiring, symbols turned on the walls, no room rectangles); `tests/test_raster_detection.py` reads that
plan printed and scanned; `tests/data/catalog.json` is a copy of the Solution Packages data.

## Limits

* DWG, HEIC photos, tiny pictures (< 500 px) and password-protected PDFs are refused with the reason.
* Scans and pictures are read from the image: symbols must look like their legend symbol at the same size;
  very faint or blurry drawings, photos taken at an angle, and symbols drawn over thick hatching read poorly.
* DXF symbols must be blocks (exploded symbols can't be counted); rooms need a closed outline and a name.
* PDF symbols are matched by the shape of the legend symbol - less certain than DXF (confidence shows it).
  Symbols drawn touching each other merge into one shape and are not counted; check with `--overlay`.
* Printed PDFs (no text) are read with OCR: check room names and counts with `--overlay`.
* PDF rooms need their walls drawn (lines or solid) and a room name inside; corridors narrower than ~4 ft clear can be
  closed off when door openings are closed.
