# Setting up PGVector Extension in Postgres Database

```bash
# Updating the package management repository details
apt update
# Installing necessary tools required to combile pgvector extension
apt install -y build-essential
# Installing posgtresql-server dev source code
apt install -y postgresql-server-dev-17
# Installing git version control software
apt install -y git
#
mkdir software-download && cd software-download 
git clone https://github.com/pgvector/pgvector.git;
cd pgvector
make
make install
```

Now we need to log into the postgres database using the `psql` command line utility

```bash
psql -U postgres
```
The default database assigned to the user `postgres` is `postgres`. Hence we need to connect to the `mylibrary` database. 

Let us list down all the available *databases*

```SQL
\list
\c mylibrary
```

Let us list down all the available `schemas`. 

```SQL
\dn
```

We need to select the *schema dev* and list down all the available *tables*. 

```SQL
SET search_path to dev;
\dt
```

Then let us list down all the currently installed database extensions. 

```bash
\dx
```

Now let's create the extension with below `CREATE EXTENSION` statement. Then list down the installed extensions again to verify. 

```sql
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA pg_catalog;
\dx
```

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



