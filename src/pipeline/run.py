# The idea is the following:
# 1) Create pipelines factory
# 2) Each pipeline assesses one of key aspects of retrieval system:
#   - Assess index building time
#   - Assess average/total index search time
#   - !!! Assess quality of retrieval: compare an index with BruteForce approach. 
#     Build plots with recall@k and retrieval time deps.