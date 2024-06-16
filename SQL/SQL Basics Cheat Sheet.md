


## The different dialects of SQL 

* MySQL
* SQLite
* SQL Server
* Oracle SQL
* PostgreSQL

Sample Data

|        |          |             |                     |                 |
| ------ | -------- | ----------- | ------------------- | --------------- |
| **id** | **city** | **country** | **number_of_rooms** | **year_listed** |
| 1      | Paris    | France      | 5                   | 2018            |
| 2      | Tokyo    | Japan       | 2                   | 2017            |
| 3      | New York | USA         | 2                   | 2022            |

## Querying tables
```sql
SELECT * 
FROM airbnb_listings;
```

## Return the city column for the table 

```sql
SELECT city 
FROM airbnb_listings;
```
## Get the city and year_listed columns from the table 
```sql
SELECT city, year_listed
FROM airbnb_listings;

```

## Get the listing id, city, orderered by the number_of_rooms ascending order

```sql
SELECT city, year_listed 
FROM airbnb_listings 
ORDER BY number_of_rooms ASC
```
## Get the listing id, city, ordered by the name_of_rooms in descending order
```sql
SELECT city, year_listed 
FROM airbnb_listings 
ORDER BY number_of_rooms DESC;

```
## Get the first 5 rows from airbnb_listings
```sql 
SELECT *
FROM airbnb_listings
LIMIT 5;
```
## Get a unique list of cities where there are listings
```sql
SELECT DISTINCT city
FROM airbnb_listings;

```
# Filtering on numeric columns
## Get all the listings where number_of_rooms is more or equal to 3

```sql
SELECT *
FROM airbnb_listings 
WHERE number_of_rooms >= 3;
```
## Get all the listings where number_of_rooms is more than 3 
```sql
SELECT *
FROM airbnb_listings 
WHERE number_of_rooms > 3;
```
## Get all the listings where number_of_rooms is lower or equal than 3

```sql 
SELECT *
FROM airbnb_listings 
WHERE number_of_rooms <= 3;
```

## Filtering columns with a range -- Get all the listing with 3 to 6 rooms
```sql
SELECT *
FROM airbnb_listings
WHERE number_of_rooms BETWEEN 3 AND 6;
```
# Filtering on text columns

## Get all the listings that are based in 'Paris'
```sql
SELECT * 
FROM airbnb_listings 
WHERE city = 'Paris';

```
## Filter one column on many conditions -- Get the listings based in the 'USA' and in 'France'

```sql
SELECT *
FROM airbnb_listings 
WHERE country IN ('USA', 'France');
```
## Get all listings where city starts with "j" and where it does not end with "t"
```sql
SELECT *
FROM airbnb_listings
%% // using regex // %%
WHERE city LIKE 'j%' AND city NOT LIKE '%t'; 
```
# Filtering on multiple columns
## Get all the listings in "Paris" where number_of_rooms is bigger than 3 
```sql
SELECT *
FROM airbnb_listings
WHERE city = 'Paris' AND number_of_rooms > 3;
```

### Get all the listings in `"Paris"` OR the ones that were listed after 2012

```sql
SELECT * 
FROM airbnb_listings
WHERE city = 'Paris' OR year_listed > 2012;
```
## Filtering on missing data\
### Get all the listings where `number_of_rooms` is missing
```sql
SELECT *
FROM airbnb_listings 
WHERE number_of_rooms IS NULL; 

```
### Get all the listings where `number_of_rooms` is not missing
```sql
SELECT *
FROM airbnb_listings 
WHERE number_of_rooms IS NOT NULL; 

```
## Simple aggregations
### Get the total number of rooms available across all listings
```sql
SELECT SUM(number_of_rooms) 
FROM airbnb_listings; 
```
### Get the average number of rooms per listing across all listings
```sql
SELECT AVG(number_of_rooms) 
FROM airbnb_listings;
```
### Get the listing with the highest number of rooms across all listings
```sql
SELECT MAX(number_of_rooms)  
FROM airbnb_listings;
```
### Get the listing with the lowest number of rooms across all listings
```sql
SELECT MIN(number_of_rooms) 
FROM airbnb_listings;
```
## Grouping, filtering, and sorting
### Get the total number of rooms for each country
```sql
SELECT country, SUM(number_of_rooms)
FROM airbnb_listings
GROUP BY country;
```
### Get the average number of rooms for each country
```sql
SELECT country, AVG(number_of_rooms)
FROM airbnb_listings
GROUP BY country;
```
### Get the listing with the maximum number of rooms for each country\
```sql
SELECT country, MAX(number_of_rooms)
FROM airbnb_listings
GROUP BY country;
```
### Get the listing with the lowest amount of rooms per country
```sql
SELECT country, MIN(number_of_rooms)
FROM airbnb_listings
GROUP BY country;
```
### For each country, get the average number of rooms per listing, sorted by ascending order
```sql
SELECT country, AVG(number_of_rooms) AS avg_rooms
FROM airbnb_listings
GROUP BY country
ORDER BY avg_rooms ASC;
```
### For Japan and the USA, get the average number of rooms per listing in each country
```sql
SELECT country, AVG(number_of_rooms)
FROM airbnb_listings
WHERE country IN ('USA', 'Japan');
GROUP BY country;
```
### Get the number of listings per country
```sql
SELECT country, COUNT(id) AS number_of_listings
FROM airbnb_listings
GROUP BY country;
```
### Get all the years where there were more than 100 listings per year

```sql
SELECT year_listed
FROM airbnb_listings
GROUP BY year_listed
HAVING COUNT(id) > 100;
```
