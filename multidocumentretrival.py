def retrieve_relevant_context(query, chroma_collection, top_k=4, alpha=0.4, b_max=3):
    query_embedding = embed_chunks([query])[0]

    # Baseline top-k retrieval (unchanged from Listing 3.2)
    base_results = chroma_collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = base_results["documents"][0]
    retrieved_meta = base_results["metadatas"][0]
    retrieved_scores = [1 - d for d in base_results["distances"][0]]  # cosine similarity

    represented_sources = {m["source"] for m in retrieved_meta}
    tau = min(retrieved_scores)
    top_score = max(retrieved_scores)
    delta = alpha * (top_score - tau)

    # Pull a larger candidate pool to search for backfill candidates
    candidate_pool = chroma_collection.query(
        query_texts=[query],
        n_results=top_k * 5,
        include=["documents", "metadatas", "distances"]
    )

    # Best candidate per unrepresented source
    best_per_source = {}
    for doc, meta, dist in zip(
        candidate_pool["documents"][0],
        candidate_pool["metadatas"][0],
        candidate_pool["distances"][0],
    ):
        source = meta["source"]
        score = 1 - dist
        if source in represented_sources:
            continue
        if source not in best_per_source or score > best_per_source[source][1]:
            best_per_source[source] = (doc, score, meta)

    # Admit candidates within the relaxation margin, strongest first, capped at b_max
    qualifying = [
        (doc, score, meta) for doc, score, meta in best_per_source.values()
        if score >= tau - delta
    ]
    qualifying.sort(key=lambda x: x[1], reverse=True)

    for doc, score, meta in qualifying[:b_max]:
        retrieved_chunks.append(doc)
        retrieved_meta.append(meta)
        retrieved_scores.append(score)

    return retrieved_chunks, retrieved_meta