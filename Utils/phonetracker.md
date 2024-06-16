
```python 
import phonenumbers

# geocoder: to know the location of the phone number
# carrier: to know the name of the carrier
# userinfo: to know the information about the phone number

from phonenumbers import geocoder, carrier, 


phonenumber = input("Enter the phone number with country code: ")
phone_number = phonenumbers.parse(phonenumber)

print(geocoder.description_for_number(phone_number, 'en'))
print(carrier.name_for_number(phone_number, 'en'))

```
not supported in many regions
