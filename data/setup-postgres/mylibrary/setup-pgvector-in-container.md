# Setting up PGVector Extension in Postgres Database
The ***PGVector*** database extension does not come pre-installed with the default *postgres* installation. We need to separately install and configure it. 

Let's now download the *pgvector database extension* and compile it and install it into our *postgres database* instance. 

```bash
# Updating the package management repository details
apt update
# Installing necessary tools required to compile pgvector extension
apt install -y build-essential
# Installing posgtresql-server dev source code
apt install -y postgresql-server-dev-17
# Installing git version control software
apt install -y git
# create a new folder to download and compile the pgvector database extension.
mkdir software-download && cd software-download 
# cloning the database extension repository
git clone https://github.com/pgvector/pgvector.git;
# going inside the cloned repository
cd pgvector
# compiling it
make
# installing the database extension.
make install
```

Now we need to log into the postgres database using the `psql` command line utility

```bash
# Using the psql command line utility to connect to the postgres database
psql -U postgres
```
The default database assigned to the user `postgres` is `postgres`. Hence we need to connect to the `mylibrary` database. 

Let us list down all the available *databases*

```SQL
-- listing the available database
\list
-- connecting to mylibrary database
\c mylibrary
```

Let us list down all the available `schemas`. 

```SQL
-- Listing the available database schemas
\dn
```

We need to select the *schema dev* and list down all the available *tables*. 

```SQL
-- selecting the dev schema
SET search_path to dev;
-- listing the tables 
\dt
```

Then let us list down all the currently installed database extensions. 

```SQL
-- Listing installed database extensions
\dx
```

Now let's create the extension with below `CREATE EXTENSION` statement. Then list down the installed extensions again to verify. 

```SQL
-- Creating the vector extension
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA pg_catalog;
-- Listing the installed database extensions to verify
\dx
```
Earlier when we created the `dev.books` table to store our books data set, we didn't have a column to hold the *vectorized_title* of the book. Therefore we now need to create a new *table column* with the type of *vector* to store the *vectorized_title*. 

Let us describe the current `schema or structure` of the table `dev.books`. 

```SQL
\dt

SELECT column_name, data_type, character_maximum_length FROM information_schema.columns WHERE table_schema = 'dev' AND table_name = 'books';
```

Now we can add another column to the existing table to keep the *vectorized book titles*. 

```sql
ALTER TABLE mylibrary.dev.books
ADD COLUMN vectorized_title vector(768);
```

Let us look at the new `schema or the structure` of the `dev.books` table. 

```SQL
SELECT column_name, data_type, character_maximum_length FROM information_schema.columns WHERE table_schema = 'dev' AND table_name = 'books';
```
We can see the new column `vectorized_title` is added with a type of `vector`. 

NOTE
Alternatively we can use below command to describe a table.That shows the number of `dimensions` expected under the `vectorized_title` column. 

```SQL
\d dev.books
```



