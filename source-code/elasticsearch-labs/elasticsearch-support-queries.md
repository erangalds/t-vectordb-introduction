# Supporting and Trouble Shooting Queries for the Elasticsearch Lab

## Enabling Enterprise License Trial
First query the currently enabled license and the available features list

```bash
curl -X GET "http://localhost:9200/_xpack"?pretty
```

If the *ml* feature is not `available`. Then we need to enable the ***Elastic Enterprise Trial*** which is valid for 30 days. 

We need to enable the `elastic` ***Enterprise Trial*** in the `docker container instance` using the `elastic api`. 

```bash
# if we are enabling it from the docker host command line
curl -X POST "http://localhost:9200/_license/start_trial?acknowledge=true"

# If we are enabling it from out dev-coding container 
curl -X POST "http://vectordb-lab-elastic:9200/_license/start_trial?acknowledge=true"

```

Then we need to re-check whether the *Machine Learning Feature* is *enabled*. We can do that with below API call. 

```bash
curl -X GET "http://localhost:9200/_xpack"?pretty
```


