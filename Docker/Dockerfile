#This launches:
#Neo4j Browser on http://localhost:7474
#Bolt (DB connection) on port 7687
#Username: neo4j | Password: password

docker run \
  --name neo4j-biograph \
  -p7474:7474 -p7687:7687 \
  -d \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.14
