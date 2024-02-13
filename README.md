Created query language capable of handling both structured and unstructured data with hash-based join and aggregation
Implemented custom lexer, parser and interpreter,  and optimized memory management by chunk-based data processing

SQL keywords and its synonyms:
Select - extract
* - $
where - when 
Group by - Categorize by
like - similar to
order - rank
limit - bound
offset - slice
From - Using

This documents the steps to run the code to execute the new query language for SQL and NoSQL databases. 


Mapping
This is the mapping for our custom query language:

Step 1: From command line, run:
python main_wrapper.py 


Step 2: You will be prompted to enter ‘sql’ or ‘nosql’ depending on your requirement

Step 3: You can either perform CRUD operations or data modification 

Here are some sample commands:

SQL:
CRUD:
INSERT table VALUES [{'sepalLength': 5.61, 'sepalWidth': 3.55, 'petalLength': 1.14, 'petalWidth': 10.2, 'variety': 'Sukuna'},{'sepalLength': 5.61, 'sepalWidth': 3.55, 'petalLength': 1.14, 'petalWidth': 10.2, 'variety': 'Moja'}]
UPDATE table VALUES {'variety': 'Roja'} CONDITION {'variety': ('==','Moja')}
DELETE table  CONDITION {'variety': ('==','Sukuna')}
Data modification:
EXTRACT $ USING data CATEGORIZE BY variety AGGREGATE BY MAX sepalLength MIN sepalWidth
EXTRACT $ USING data LEFT JOIN sepalLength = sL RANK sepalLength DESC 
EXTRACT $ USING data RIGHT JOIN sepalLength = sL RANK sepalLength DESC 
EXTRACT $ USING data WHEN sepalLength EQ 6 LEFT JOIN sepalLength = sL RANK sepalLength DESC BOUND 3
EXTRACT $ USING data PROJECT petalWidth sepalLength variety RANK petalWidth 
EXTRACT $ USING data WHEN sepalLength LT 6 AND petalLength GT 3 AND variety EQ 'Mojito'


NoSQL:
CRUD:
Insertion: INSERT VALUES {"sepalLength": "5.6", "sepalWidth": "3.2", "petalLength": "5.0", "petalWidth": "1.8", "species": "setosa"}
Deletion: DELETE WHERE species = setosa
Updating: UPDATE VALUE WHERE species = setosa {“sepalLength”: 5.6}

Data modification:
NoSQL:
EXTRACT $ USING data PROJECT species,sepalLength
EXTRACT $ USING data BOUND 2
EXTRACT $ USING data WHERE sepalLength GT 5.0 AND sepalLength LE 7.4
EXTRACT $ USING data DISTINCT species PROJECT species
EXTRACT $ USING data AVERAGE sepalLength
EXTRACT $ USING data RANK sepalLength PROJECT sepalLength,species
EXTRACT $ USING data CATEGORIZE BY species
EXTRACT $ USING data MIN sepalLength CATEGORIZE BY species


Note: If you wish to change the dataset, navigate to wrapper_nosql.py under nosql directory and change the variable ‘json_file_path’ in the 7th line. Try changing path in the same line in case the code doesn’t run.
