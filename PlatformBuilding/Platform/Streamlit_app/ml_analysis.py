import json

import pandas as pd
from py2neo import Graph, NodeMatcher

class MLAnalysis:
    def __init__(self, graph_name='myGraph', embedding_dim=8, link_prediction_pipeline='myPipe'):
        self.graph_name = graph_name
        self.embedding_dim = embedding_dim
        self.link_prediction_pipeline = link_prediction_pipeline
        self.graph = Graph(host="0.0.0.0", port="7687", auth=('neo4j', 'your_password'))

    def create_new_relationships(self, property='strength_IF'):
        # Create negative and positive strengths
        query = """
        MATCH (n1:Microbe)-[r:STRENGTH]->(n2:Disease) 
        WHERE r.{} > 0
        MERGE (n1)-[new_r:STRENGTH_POSITIVE]->(n2)
        SET new_r.strength = r.{} 
        """.format(property, property)
        self.graph.run(query)

        query = """
                MATCH (n1:Microbe)-[r:STRENGTH]->(n2:Disease) 
                WHERE r.{} < 0
                MERGE (n1)-[new_r:STRENGTH_NEGATIVE]->(n2)
                SET new_r.strength = - r.{} 
                """.format(property, property)
        self.graph.run(query)

        query = """
            MATCH (n1)-[r:PARENT]->(n2) 
            SET r.strength = 1
            """.format(property, property)
        self.graph.run(query)


    def project_graph(self):
        query = """
        CALL gds.graph.project(
      '{}', """.format(self.graph_name)

        query_continuation = """
        {
            Microbe: {properties: ['metapath-embedding']},
            Disease: {properties: ['metapath-embedding']}
        }, 
        {
            STRENGTH_POSITIVE: {orientation: 'UNDIRECTED', properties: ['strength']},
            STRENGTH_NEGATIVE: {orientation: 'UNDIRECTED', properties: ['strength']},
            PARENT: {orientation: 'NATURAL', properties: ['strength']}
        })
        YIELD
          graphName AS graph,
          relationshipProjection AS readProjection,
          nodeCount AS nodes,
          relationshipCount AS rels
        """

        query = query + query_continuation
        result = self.graph.run(query)


    def create_embeddings(self, algorithm='fastRP'):
        if algorithm == 'fastRP':
            query = """
                CALL gds.fastRP.mutate(
                  '{}',""".format(self.graph_name)
            query_continuation = """
                  |-
                    embeddingDimension: {},
                    mutateProperty: 'fastRP-embedding',
                    relationshipWeightProperty: 'strength'
                  -|
                )
                YIELD nodePropertiesWritten
            """.format(self.embedding_dim).replace('|-', '{').replace('-|', '}')
            query = query + query_continuation
            result = self.graph.run(query)

        elif algorithm == 'node2vec':
            query = """
                    CALL gds.fastRP.mutate(
                    '{}',""".format(self.graph_name)
            query_continuation = """
                              |-
                                embeddingDimension: {},
                                mutateProperty: 'node2vec-embedding',
                                relationshipWeightProperty: 'strength'
                              -|
                            )
                            YIELD nodePropertiesWritten
                        """.format(self.embedding_dim).replace('|-', '{').replace('-|', '}')
            query = query + query_continuation
            result = self.graph.run(query)


        else:# algorithm == 'graphSAGE':
            # Create numeric property
            query = """
                CALL gds.degree.mutate(
                  '{}',   
            """.format(self.graph_name)
            query_continuation = """
            {
                mutateProperty:'degree'
            }   
                )
            """
            query = query + query_continuation
            result = self.graph.run(query)

            # First train the graph
            query = """
            CALL gds.beta.graphSage.train(
              '{}',
            """.format(self.graph_name)
            query_continuation = """
            |-
                modelName: 'sage_model',
                featureProperties:['degree'],
                relationshipWeightProperty: 'strength',
                embeddingDimension:{},
                activationFunction:'sigmoid',
                batchSize:32,
                learningRate:0.01,
                maxIterations:50,
                epochs:10
              -|
            )
            """.format(self.embedding_dim).replace('|-', '{').replace('-|', '}')
            query = query + query_continuation
            result = self.graph.run(query)

            # Second, write results to graph
            query = """
            CALL gds.beta.graphSage.mutate(
              '{}',
            """.format(self.graph_name)

            query_continuation = """
            {
                mutateProperty: 'sage-embedding',
                modelName: 'sage_model'
              }
            ) YIELD
              nodeCount,
              nodePropertiesWritten
            """
            query = query + query_continuation
            result = self.graph.run(query)

    def write_embeddings(self, embedding_property='fastRP-embedding'):
        query_pos = """
        CALL gds.graph.nodeProperties.write('{}', ['{}'])
            YIELD propertiesWritten
        """.format(self.graph_name, embedding_property)
        result = self.graph.run(query_pos)


    def find_nearest_neighbors(self, embedding_property='fastRP-embedding', label='Microbe', cui='C0242946'):
        # Find the id of the node
        query = """
        MATCH (m:{})
        WHERE m.cui = '{}'
        RETURN id(m) AS sourceNodeId 
        """.format(label, cui)
        source_node = self.graph.run(query).data()[0]['sourceNodeId']

        # Run the algorithm in stream mode
        query = """
        CALL gds.knn.filtered.stream('{}'
        """.format(self.graph_name)
        query += ', {'
        query_continuation = """
            topK: 10,
            nodeProperties: ['{}'],
            sourceNodeFilter: {},
            targetNodeFilter: '{}',
            sampleRate: 1.0
        """.format(embedding_property, source_node, label)

        query_continuation2 = """
        })
        YIELD node1, node2, similarity
        RETURN gds.util.asNode(node1).name AS MicrobeSource, gds.util.asNode(node1).cui AS MicrobeSourceCui, 
        gds.util.asNode(node2).name AS MicrobeTarget, gds.util.asNode(node2).cui AS MicrobeTargetCui, similarity
        ORDER BY similarity DESCENDING
        """
        query = query + query_continuation + query_continuation2
        result = self.graph.run(query).to_data_frame()
        return result

    def get_clusters(self, embedding_property='fastRP-embedding', n_clusters=5):
        query = ("""
        CALL gds.kmeans.mutate('{}', |-
          nodeLabels:['Microbe', 'Disease'],
          relationshipTypes:['PARENT', 'STRENGTH_NEGATIVE', 'STRENGTH_POSITIVE'],
          nodeProperty: '{}',
          k: {},
          maxIterations: 100,
          numberOfRestarts:10,
          randomSeed: 42,
          mutateProperty: '{}-kmeans{}'
        -|)
        YIELD communityDistribution
        """.format(self.graph_name, embedding_property, n_clusters, embedding_property.split('-')[0], n_clusters)
                 .replace('|-', '{').replace('-|', '}'))

        result = self.graph.run(query)


    def write_clusters(self, kmeans_property='fastRP-kmeans'):
        query_pos = """
                CALL gds.graph.nodeProperties.write('{}', ['{}'])
                    YIELD propertiesWritten
                """.format(self.graph_name, kmeans_property)
        result = self.graph.run(query_pos)

    def train_link_prediction(self, embedding_property='fastRP-embedding'):
        link_prediction_pipeline = self.link_prediction_pipeline

        # Pipeline creation
        query = """
        CALL gds.beta.pipeline.linkPrediction.create('{}')
        """.format(link_prediction_pipeline)
        result = self.graph.run(query)

        # Add precomputed features
        query = """
        CALL gds.beta.pipeline.linkPrediction.addFeature('{}', 'cosine', |-
          nodeProperties: ['{}']
        -|) YIELD featureSteps
        """.format(link_prediction_pipeline, embedding_property).replace('|-', '{').replace('-|', '}')
        result = self.graph.run(query)

        # Configure split
        query = """
        CALL gds.beta.pipeline.linkPrediction.configureSplit('{}', |-
          testFraction: 0.2,
          trainFraction: 0.7,
          validationFolds: 5
        -|)
        YIELD splitConfig
        """.format(link_prediction_pipeline).replace('|-', '{').replace('-|', '}')
        results = self.graph.run(query)

        # Adding models
        query_regression = """
        CALL gds.beta.pipeline.linkPrediction.addLogisticRegression('{}')
            YIELD parameterSpace
        """.format(link_prediction_pipeline)

        query_randomForest = """
                CALL gds.beta.pipeline.linkPrediction.addRandomForest('{}', |-numberOfDecisionTrees: 10-|)
                    YIELD parameterSpace
                """.format(link_prediction_pipeline).replace('|-', '{').replace('-|', '}')

        query_MLP = """
                CALL gds.alpha.pipeline.linkPrediction.addMLP('{}')
                    YIELD parameterSpace
                """.format(link_prediction_pipeline)

        self.graph.run(query_regression)
        self.graph.run(query_randomForest)
        self.graph.run(query_MLP)

        # Train positive model
        query_pos = """
        CALL gds.beta.pipeline.linkPrediction.train('{}', |-
          pipeline: '{}',
          modelName: '{}',
          metrics: ['AUCPR', 'OUT_OF_BAG_ERROR'],
          sourceNodeLabel: 'Microbe',
          targetNodeLabel: 'Disease',
          targetRelationshipType: 'STRENGTH_POSITIVE',
          randomSeed: 12
        -|) YIELD modelInfo, modelSelectionStats
        RETURN
          modelInfo.bestParameters AS winningModel,
          modelInfo.metrics.AUCPR.train.avg AS avgTrainScore,
          modelInfo.metrics.AUCPR.outerTrain AS outerTrainScore,
          modelInfo.metrics.AUCPR.test AS testScore,
          [cand IN modelSelectionStats.modelCandidates | cand.metrics.AUCPR.validation.avg] AS validationScores
        """.format(self.graph_name, link_prediction_pipeline,
                   link_prediction_pipeline + '_model_pos').replace('|-', '{').replace('-|', '}')
        result = self.graph.run(query_pos)

        # Train Negative model
        query_neg = """
               CALL gds.beta.pipeline.linkPrediction.train('{}', |-
                 pipeline: '{}',
                 modelName: '{}',
                 metrics: ['AUCPR', 'OUT_OF_BAG_ERROR'],
                 sourceNodeLabel: 'Microbe',
                 targetNodeLabel: 'Disease',
                 targetRelationshipType: 'STRENGTH_NEGATIVE',
                 randomSeed: 12
               -|) YIELD modelInfo, modelSelectionStats
               RETURN
                 modelInfo.bestParameters AS winningModel,
                 modelInfo.metrics.AUCPR.train.avg AS avgTrainScore,
                 modelInfo.metrics.AUCPR.outerTrain AS outerTrainScore,
                 modelInfo.metrics.AUCPR.test AS testScore,
                 [cand IN modelSelectionStats.modelCandidates | cand.metrics.AUCPR.validation.avg] AS validationScores
               """.format(self.graph_name, link_prediction_pipeline,
                          link_prediction_pipeline + '_model_neg').replace('|-', '{').replace('-|', '}')
        result = self.graph.run(query_neg)


    def link_prediction(self):
        link_prediction_pipeline = self.link_prediction_pipeline


        query_pos = ("""
        CALL gds.beta.pipeline.linkPrediction.predict.mutate('{}', |-
          modelName: '{}',
          relationshipTypes: ['STRENGTH_POSITIVE'],
          mutateRelationshipType: 'STRENGTH_POSITIVE_PRED',
          mutateProperty:'pred_strength',
          sampleRate: 0.5,
          topK: 10,
          randomJoins: 10,
          sourceNodeLabel: 'Microbe',
          targetNodeLabel: 'Disease',
          maxIterations: 100,
          concurrency: 4
          //threshold: 0.5
        -|)
         YIELD relationshipsWritten, samplingStats
         //YIELD node1, node2, probability
         //RETURN gds.util.asNode(node1).name AS Microbe, gds.util.asNode(node2).name AS Disease, probability
         //ORDER BY probability DESC
        """.format(self.graph_name, link_prediction_pipeline + '_model_pos')
                     .replace('|-', '{').replace('-|', '}'))
        result = self.graph.run(query_pos)

        query_neg = ("""
                CALL gds.beta.pipeline.linkPrediction.predict.mutate('{}', |-
                  modelName: '{}',
                  relationshipTypes: ['STRENGTH_NEGATIVE'],
                  mutateRelationshipType: 'STRENGTH_NEGATIVE_PRED',
                  mutateProperty:'pred_strength',
                  sampleRate: 0.5,
                  topK: 10,
                  randomJoins: 10,
                  sourceNodeLabel: 'Microbe',
                  targetNodeLabel: 'Disease',
                  maxIterations: 100,
                  concurrency: 4
                  //threshold: 0.5
                -|)
                 YIELD relationshipsWritten, samplingStats
                 //YIELD node1, node2, probability
                 //RETURN gds.util.asNode(node1).name AS Microbe, gds.util.asNode(node2).name AS Disease, probability
                 //ORDER BY probability DESC
                """.format(self.graph_name, link_prediction_pipeline + '_model_neg')
                     .replace('|-', '{').replace('-|', '}'))
        result = self.graph.run(query_neg)


    def shortest_path(self, node1, node2):
        query = ("""
        MATCH (source:{} |-cui: '{}'-|), (target:{} |-cui: '{}'-|)
            CALL gds.shortestPath.dijkstra.stream('{}', |-
                sourceNode: source,
                targetNode: target,
                relationshipTypes: ['STRENGTH_POSITIVE', 'STRENGTH_NEGATIVE', 'PARENT']
            -|)
            YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
            RETURN
                index,
                gds.util.asNode(sourceNode).name AS sourceNodeName,
                gds.util.asNode(targetNode).name AS targetNodeName,
                totalCost,
                [nodeId IN nodeIds | gds.util.asNode(nodeId).name] AS nodeNames,
                [nodeId IN nodeIds | gds.util.asNode(nodeId).cui] AS nodecuis
                //costs,
                //nodes(path) as path
            ORDER BY index
        """.format(node1['label'], node1['cui'], node2['label'], node2['cui'], self.graph_name)
                 .replace('|-', '{').replace('-|', '}'))
        result = self.graph.run(query).to_data_frame()
        cuis = result['nodecuis'].values[0]

        all_results = []
        for i in range(len(cuis) - 1):
            query = """
            Match (n)-[r:PARENT|STRENGTH]-(m)
            Where n.cui='{}' AND m.cui='{}'
            RETURN n.name as Source, labels(n)[0] as SourceType, type(r) as Relation, r.strength_raw as Strength, m.name as Target, labels(m)[0] as TargetType
            """.format(cuis[i], cuis[i+1])
            result = self.graph.run(query).to_data_frame()
            all_results.append(result)
        all_results = pd.concat(all_results)
        return all_results

    def write_link_predictions(self):
        query_pos = """
        CALL gds.graph.relationship.write(
          '{}',
          'STRENGTH_POSITIVE_PRED',
          'pred_strength'
        )
        YIELD
          graphName, relationshipType, relationshipProperty, relationshipsWritten, propertiesWritten
        """.format(self.graph_name)
        result = self.graph.run(query_pos)
        print(result)

        query_neg = """
        CALL gds.graph.relationship.write(
          '{}',
          'STRENGTH_NEGATIVE_PRED',
          'pred_strength'
        )
        YIELD
          graphName, relationshipType, relationshipProperty, relationshipsWritten, propertiesWritten
        """.format(self.graph_name)
        result = self.graph.run(query_neg)
        print(result)

        # Delete one of the directions
        query_delete = """
        MATCH (:Disease)-[r:STRENGTH_NEGATIVE_PRED|STRENGTH_POSITIVE_PRED]->(:Microbe)
        DELETE r
        """
        self.graph.run(query_delete)

        # Delete if there is some duplicates
        query_delete = """
        MATCH (n1)-[r:STRENGTH_NEGATIVE_PRED|STRENGTH_POSITIVE_PRED]->(n2)
            WITH n1, n2, type(r) AS t, count(*) AS count 
            WHERE count > 1
            MATCH (n1)-[r2]->(n2) 
            WHERE type(r2) = t
            DELETE r2
        """
        self.graph.run(query_delete)

        # Delete all relationships in which the strength is less than 0.5

        query_delete = """
            MATCH (n1)-[r:STRENGTH_NEGATIVE_PRED|STRENGTH_POSITIVE_PRED]-(n2)
            WHERE r.pred_strength < 0.5
            DELETE r
            """
        self.graph.run(query_delete)


    def drop_relations(self):
        query = """
               MATCH (n1)-[r:STRENGTH_NEGATIVE_PRED|STRENGTH_POSITIVE_PRED|STRENGTH_NEGATIVE|STRENGTH_POSITIVE]-(n2)
               DELETE r 
               """
        self.graph.run(query)


    def drop_graph(self):
        query = """
        CALL gds.graph.drop('{}') YIELD graphName;
        """.format(self.graph_name)
        try:
            self.graph.run(query)
        except Exception as e:
            print(e)

        query = """
        CALL gds.model.drop('sage_model')
            YIELD modelName, modelType, modelInfo, loaded, stored, published
        """
        try:
            self.graph.run(query)
        except Exception as e:
            print(e)

        query = """
               CALL gds.model.drop('{}')
                   YIELD modelName, modelType, modelInfo, loaded, stored, published
               """.format(self.link_prediction_pipeline + '_model_pos')
        try:
            self.graph.run(query)
        except Exception as e:
            print(e)

        query = """
                CALL gds.model.drop('{}')
                YIELD modelName, modelType, modelInfo, loaded, stored, published
            """.format(self.link_prediction_pipeline + '_model_neg')
        try:
            self.graph.run(query)
        except Exception as e:
            print(e)

        query = """
        CALL gds.pipeline.drop('{}')
            YIELD pipelineName, pipelineType
        """.format(self.link_prediction_pipeline)
        try:
            self.graph.run(query)
        except Exception as e:
            print(e)

    def drop_properties(self, property='fastRP-embedding'):
        query = """
        MATCH (n:Microbe|Disease)
        REMOVE n.`{}`
        RETURN n
        """.format(property)
        self.graph.run(query)

    def get_embeddings(self, embedding_property='fastRP-embedding', label='Microbe'):
        query = """
        MATCH (n:{} )
        RETURN n.name as name, n.`{}` as embedding
        """.format(label, embedding_property)
        results = self.graph.run(query).to_data_frame()
        return results




if __name__ == '__main__':
    ml = MLAnalysis(graph_name='myGraph', embedding_dim=64, link_prediction_pipeline='myPipe')

    ml.drop_graph()
    ml.drop_relations()
    ml.drop_properties(property='fastRP-embedding')
    ml.drop_properties(property='node2vec-embedding')
    ml.drop_properties(property='sage-embedding')

    ml.drop_properties(property='fastRP-kmeans3')
    ml.drop_properties(property='node2vec-kmeans3')
    ml.drop_properties(property='sage-kmeans3')

    ml.drop_properties(property='fastRP-kmeans4')
    ml.drop_properties(property='node2vec-kmeans4')
    ml.drop_properties(property='sage-kmeans4')

    ml.drop_properties(property='fastRP-kmeans5')
    ml.drop_properties(property='node2vec-kmeans5')
    ml.drop_properties(property='sage-kmeans5')

    ml.drop_properties(property='fastRP-kmeans6')
    ml.drop_properties(property='node2vec-kmeans6')
    ml.drop_properties(property='sage-kmeans6')

    ml.drop_properties(property='fastRP-kmeans7')
    ml.drop_properties(property='node2vec-kmeans7')
    ml.drop_properties(property='sage-kmeans7')

    print('Creating Embeddings...')
    ml.create_new_relationships(property='strength_IF')
    ml.project_graph()
    ml.create_embeddings(algorithm='fastRP')
    ml.create_embeddings(algorithm='node2vec')
    ml.create_embeddings(algorithm='sage')
    ml.write_embeddings(embedding_property='fastRP-embedding')
    ml.write_embeddings(embedding_property='node2vec-embedding')
    ml.write_embeddings(embedding_property='sage-embedding')

    # KNN
    print('KNN...')
    ml.find_nearest_neighbors(embedding_property='fastRP-embedding', label='Microbe', cui='C0242946')

    # K-means
    print('Kmeans...')
    ml.get_clusters(embedding_property='fastRP-embedding', n_clusters=3)
    ml.get_clusters(embedding_property='node2vec-embedding', n_clusters=3)
    ml.get_clusters(embedding_property='sage-embedding', n_clusters=3)
    ml.get_clusters(embedding_property='metapath-embedding', n_clusters=3)

    ml.write_clusters(kmeans_property='fastRP-kmeans3')
    ml.write_clusters(kmeans_property='node2vec-kmeans3')
    ml.write_clusters(kmeans_property='sage-kmeans3')
    ml.write_clusters(kmeans_property='metapath-kmeans3')

    ml.get_clusters(embedding_property='fastRP-embedding', n_clusters=4)
    ml.get_clusters(embedding_property='node2vec-embedding', n_clusters=4)
    ml.get_clusters(embedding_property='sage-embedding', n_clusters=4)
    ml.get_clusters(embedding_property='metapath-embedding', n_clusters=4)
    ml.write_clusters(kmeans_property='fastRP-kmeans4')
    ml.write_clusters(kmeans_property='node2vec-kmeans4')
    ml.write_clusters(kmeans_property='sage-kmeans4')
    ml.write_clusters(kmeans_property='metapath-kmeans4')


    ml.get_clusters(embedding_property='fastRP-embedding', n_clusters=5)
    ml.get_clusters(embedding_property='node2vec-embedding', n_clusters=5)
    ml.get_clusters(embedding_property='sage-embedding', n_clusters=5)
    ml.get_clusters(embedding_property='metapath-embedding', n_clusters=5)
    ml.write_clusters(kmeans_property='fastRP-kmeans5')
    ml.write_clusters(kmeans_property='node2vec-kmeans5')
    ml.write_clusters(kmeans_property='sage-kmeans5')
    ml.write_clusters(kmeans_property='metapath-kmeans5')

    ml.get_clusters(embedding_property='fastRP-embedding', n_clusters=6)
    ml.get_clusters(embedding_property='node2vec-embedding', n_clusters=6)
    ml.get_clusters(embedding_property='sage-embedding', n_clusters=6)
    ml.get_clusters(embedding_property='metapath-embedding', n_clusters=6)
    ml.write_clusters(kmeans_property='fastRP-kmeans6')
    ml.write_clusters(kmeans_property='node2vec-kmeans6')
    ml.write_clusters(kmeans_property='sage-kmeans6')
    ml.write_clusters(kmeans_property='metapath-kmeans6')


    # Link Prediction
    print('Link Prediction...')
    ml.train_link_prediction(embedding_property='fastRP-embedding')
    ml.link_prediction()
    ml.write_link_predictions()



    ml.shortest_path(node1={'label': 'Disease', 'cui':'C0036690'}, node2={'label':'Disease', 'cui': 'C5546017'})
    ml.shortest_path(node1={'label': 'Microbe', 'cui':'C0042447'}, node2={'label':'Disease', 'cui': 'C0019340'})
