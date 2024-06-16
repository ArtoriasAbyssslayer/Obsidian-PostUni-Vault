
```python
from barcode import EAN13
from barcode.writer import ImageWriter

number: str = input("Enter the number: ")
code = EAN13(number, writer=ImageWriter())
code.save("barcode")
print("Barcode generated successfully")
```
