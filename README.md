# Drawing2Quote reader for .NET

The drawing reader of Drawing2Quote, ported from Python (branch `main`, `app/extract`) to .NET 8 with the same rules
and the same results. It reads a DXF, a CAD PDF, a printed PDF (OCR), a scanned PDF or a picture into floors → areas → electrical
point counts, and returns the same JSON as the Python service's `/extract`.

| Python                 | .NET                                                                                  |
|------------------------|---------------------------------------------------------------------------------------|
| PyMuPDF                | MuPDF.NET (the same MuPDF engine, 1.28)                                               |
| shapely (GEOS)         | NetTopologySuite for geometries and tests; buffer / union / difference / intersection / simplify run in GEOS 3.14 itself (`native/`, the build shapely ships) |
| OpenCV (cv2)           | OpenCvSharp4                                                                          |
| rapidocr-onnxruntime   | ONNX Runtime with the same PP-OCRv4 models (`models/`) and the same pre/post-processing |
| ezdxf                  | ACadSharp                                                                             |
| PyYAML                 | YamlDotNet (`src/Drawing2Quote.Reader/config/block_mapping.yaml`)                      |

Python behaviour the results depend on (half-to-even rounding, float formatting, numpy's pairwise sums, medians and
percentiles in float32, difflib's ratio, stable sorts) is reproduced in `Py.cs`.

## Use

```csharp
using Drawing2Quote.Reader;

var ex = DrawingReader.Extract(fileName, bytes);   // throws RejectedFileException with the reason for the user
string json = ex.ToJson();                          // as /extract returns it
```

`Ocr.Progress = (page, pages) => ...` reports OCR progress. The OCR cache is the Python reader's
(`%TEMP%\drawing2quote-ocr`, or `D2Q_OCR_CACHE`): a drawing either reader has read is not read again.

Command line:

```
d2q plan.pdf                          what was read
d2q plan.pdf --json --out plan.json   the extraction as JSON
d2q plan.pdf --overlay checked.pdf    a copy of the drawing with rooms and symbols marked
d2q plan.pdf --pages 5-7,12           only these pages
```

64-bit Windows (MuPDF, OpenCV, ONNX Runtime and GEOS are native). MuPDF may print `warning: ...` lines on
stderr for damaged PDF content; they don't affect the result.

## Checks

```
dotnet test tests/Drawing2Quote.Reader.Tests
```

The Python suite's reading tests, and the output compared with the Python reader's for the test drawings
(`data/python_*.json`).

The Python reader can still be run against the same drawings from branch `main` to compare: both give the same
JSON (checked on 14 drawings: CAD and printed PDFs, scans, photos, DXF).

## Differences from the Python reader

Rare cases the Python reader fails on with "The drawing could not be read" are read here: a legend sheet with only
text symbols ("AC", "TV") used for other sheets, a symbol that looks the same turned 90 degrees but not 180, a traced
space that disappears once the solid walls are taken off, a wall picture with no measurable wall thickness.
