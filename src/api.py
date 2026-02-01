"""REST API for bundle recommendations."""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

from src.recommendation_engine import BundleRecommendationEngine

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global engine instance (lazy loaded)
_engine: Optional[BundleRecommendationEngine] = None


def get_engine() -> BundleRecommendationEngine:
    """Lazy load and return the recommendation engine."""
    global _engine
    if _engine is None:
        _engine = BundleRecommendationEngine()
        if not _engine.load_model("models/recommendation_engine.pkl"):
            logger.warning("Failed to load recommendation engine model")
            raise RuntimeError("Recommendation engine model not found. Train the model first.")
    return _engine


@app.route("/health", methods=["GET"])
def health() -> Tuple[Dict[str, str], int]:
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route("/api/v1/recommenders", methods=["GET"])
def get_recommenders() -> Tuple[Dict[str, Any], int]:
    """
    Get list of available recommenders and their metadata.
    
    Returns:
        {
            "status": "success",
            "recommenders": [
                {
                    "name": str,
                    "class": str,
                    "available": bool
                }
            ],
            "count": int
        }
    """
    try:
        engine = get_engine()
        recommenders_list = []
        
        for name, recommender in engine.recommenders.items():
            recommenders_list.append({
                "name": name,
                "class": type(recommender).__name__,
                "available": True
            })
        
        return jsonify({
            "status": "success",
            "recommenders": recommenders_list,
            "count": len(recommenders_list)
        }), 200
    
    except Exception as exc:
        logger.error(f"Error getting recommenders: {exc}")
        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


@app.route("/api/v1/bundles", methods=["GET"])
def get_bundles_for_product() -> Tuple[Dict[str, Any], int]:
    """
    Get bundle recommendations for a product from all available models.
    
    Query Parameters:
        product_description (str, required): Product description
        threshold (float, optional): Confidence threshold (0.0-1.0), default: 0.3
        top_n (int, optional): Number of bundles per model, default: 5
    
    Returns:
        {
            "status": "success",
            "product_description": str,
            "recommendations": [
                {
                    "recommender": str,
                    "confidence": float,
                    "bundles": [list of bundles],
                    "count": int
                }
            ],
            "ensemble_confidence": float,
            "total_models": int
        }
    """
    try:
        # Get query parameters
        product_description = request.args.get("product_description", "").strip()
        if not product_description:
            return jsonify({"status": "error", "message": "product_description parameter is required"}), 400
        
        threshold = request.args.get("threshold", 0.3, type=float)
        if not 0.0 <= threshold <= 1.0:
            return jsonify({"status": "error", "message": "threshold must be between 0.0 and 1.0"}), 400
        
        top_n = request.args.get("top_n", 5, type=int)
        if top_n < 1:
            return jsonify({"status": "error", "message": "top_n must be >= 1"}), 400
        
        # Get engine
        engine = get_engine()
        
        # Get recommendations from all available recommenders
        recommendations = []
        confidences = []
        
        for recommender_name in engine.recommenders.keys():
            try:
                recommendation = engine.recommend_bundles(
                    customer_transaction=[product_description],
                    threshold=threshold,
                    recommender_name=recommender_name
                )
                
                bundles = recommendation.get("bundles", [])[:top_n]
                confidence = recommendation.get("confidence", 0.0)
                
                recommendations.append({
                    "recommender": recommender_name,
                    "confidence": round(confidence, 3),
                    "bundles": [list(bundle) for bundle in bundles],
                    "count": len(bundles)
                })
                
                confidences.append(confidence)
            
            except Exception as exc:
                logger.warning(f"Error getting recommendations from {recommender_name}: {exc}")
                recommendations.append({
                    "recommender": recommender_name,
                    "error": str(exc),
                    "bundles": [],
                    "confidence": 0.0,
                    "count": 0
                })
        
        # Compute ensemble confidence (average)
        ensemble_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return jsonify({
            "status": "success",
            "product_description": product_description,
            "recommendations": recommendations,
            "ensemble_confidence": round(ensemble_confidence, 3),
            "total_models": len(engine.recommenders)
        }), 200
    
    except Exception as exc:
        logger.error(f"Error getting bundles: {exc}")
        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


@app.route("/api/v1/bundles/batch", methods=["POST"])
def get_bundles_batch() -> Tuple[Dict[str, Any], int]:
    """
    Get bundle recommendations for multiple products from all available models.
    
    Request Body:
        {
            "product_descriptions": [list of product descriptions],
            "threshold": 0.3,
            "top_n": 5
        }
    
    Returns:
        {
            "status": "success",
            "results": [
                {
                    "product_description": str,
                    "recommendations": [per-model recommendations],
                    "ensemble_confidence": float,
                    "total_models": int
                }
            ],
            "total": int
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Request body must be JSON"}), 400
        
        product_descriptions = data.get("product_descriptions", [])
        if not isinstance(product_descriptions, list) or len(product_descriptions) == 0:
            return jsonify({"status": "error", "message": "product_descriptions must be a non-empty list"}), 400
        
        threshold = data.get("threshold", 0.3)
        if not 0.0 <= threshold <= 1.0:
            return jsonify({"status": "error", "message": "threshold must be between 0.0 and 1.0"}), 400
        
        top_n = data.get("top_n", 5)
        if top_n < 1:
            return jsonify({"status": "error", "message": "top_n must be >= 1"}), 400
        
        # Get engine
        engine = get_engine()
        
        # Compute recommendations for each product
        results = []
        for product_description in product_descriptions:
            try:
                product_description = str(product_description).strip()
                if not product_description:
                    continue
                
                # Get recommendations from all recommenders
                recommendations = []
                confidences = []
                
                for recommender_name in engine.recommenders.keys():
                    try:
                        recommendation = engine.recommend_bundles(
                            customer_transaction=[product_description],
                            threshold=threshold,
                            recommender_name=recommender_name
                        )
                        
                        bundles = recommendation.get("bundles", [])[:top_n]
                        confidence = recommendation.get("confidence", 0.0)
                        
                        recommendations.append({
                            "recommender": recommender_name,
                            "confidence": round(confidence, 3),
                            "bundles": [list(bundle) for bundle in bundles],
                            "count": len(bundles)
                        })
                        
                        confidences.append(confidence)
                    
                    except Exception as exc:
                        logger.warning(f"Error from {recommender_name} for {product_description}: {exc}")
                        recommendations.append({
                            "recommender": recommender_name,
                            "error": str(exc),
                            "bundles": [],
                            "confidence": 0.0,
                            "count": 0
                        })
                
                ensemble_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                results.append({
                    "product_description": product_description,
                    "recommendations": recommendations,
                    "ensemble_confidence": round(ensemble_confidence, 3),
                    "total_models": len(engine.recommenders)
                })
            
            except Exception as exc:
                logger.warning(f"Error processing product {product_description}: {exc}")
                results.append({
                    "product_description": str(product_description),
                    "error": str(exc),
                    "recommendations": [],
                    "total_models": 0
                })
        
        return jsonify({
            "status": "success",
            "results": results,
            "total": len(results)
        }), 200
    
    except Exception as exc:
        logger.error(f"Error in batch endpoint: {exc}")
        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


@app.route("/api/v1/cross-sell", methods=["GET"])
def get_cross_sell() -> Tuple[Dict[str, Any], int]:
    """
    Get cross-sell product suggestions for a product from all available models.
    
    Query Parameters:
        product_description (str, required): Product description
        top_n (int, optional): Number of suggestions per model, default: 5
    
    Returns:
        {
            "status": "success",
            "product_description": str,
            "suggestions": [
                {
                    "recommender": str,
                    "products": [
                        {"product": str, "affinity_score": float}
                    ],
                    "count": int
                }
            ],
            "total_models": int
        }
    """
    try:
        product_description = request.args.get("product_description", "").strip()
        if not product_description:
            return jsonify({"status": "error", "message": "product_description parameter is required"}), 400
        
        top_n = request.args.get("top_n", 5, type=int)
        if top_n < 1:
            return jsonify({"status": "error", "message": "top_n must be >= 1"}), 400
        
        engine = get_engine()
        
        # Get cross-sell suggestions from all recommenders
        suggestions = []
        
        for recommender_name in engine.recommenders.keys():
            try:
                cross_sell = engine.get_cross_sell_products(
                    customer_transaction=[product_description],
                    top_n=top_n,
                    recommender_name=recommender_name
                )
                
                products = [
                    {"product": product, "affinity_score": round(score, 3)}
                    for product, score in cross_sell
                ]
                
                suggestions.append({
                    "recommender": recommender_name,
                    "products": products,
                    "count": len(products)
                })
            
            except Exception as exc:
                logger.warning(f"Error getting cross-sell from {recommender_name}: {exc}")
                suggestions.append({
                    "recommender": recommender_name,
                    "error": str(exc),
                    "products": [],
                    "count": 0
                })
        
        return jsonify({
            "status": "success",
            "product_description": product_description,
            "suggestions": suggestions,
            "total_models": len(engine.recommenders)
        }), 200
    
    except Exception as exc:
        logger.error(f"Error getting cross-sell: {exc}")
        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


@app.route("/api/v1/stats", methods=["GET"])
def get_stats() -> Tuple[Dict[str, Any], int]:
    """
    Get engine statistics including bundle count and recommender info.
    
    Returns:
        {
            "status": "success",
            "stats": {
                "total_bundles": int,
                "total_recommenders": int,
                "recommenders": [list of names]
            }
        }
    """
    try:
        engine = get_engine()
        stats = engine.get_stats()
        
        # Augment stats with recommender names
        stats["recommenders"] = list(engine.recommenders.keys())
        stats["total_recommenders"] = len(engine.recommenders)
        
        return jsonify({
            "status": "success",
            "stats": stats
        }), 200
    
    except Exception as exc:
        logger.error(f"Error getting stats: {exc}")
        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# Static file serving for web UI
@app.route("/")
def serve_index():
    """Serve the main index.html page."""
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    if os.path.exists(os.path.join(web_dir, "index.html")):
        return send_file(os.path.join(web_dir, "index.html"))
    return jsonify({"status": "error", "message": "Web UI not found"}), 404


@app.route("/assets/<path:filename>")
def serve_static(filename):
    """Serve static files (JS, CSS, TSV) from web directory."""
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    file_path = os.path.join(web_dir, filename)
    
    # Security check: prevent directory traversal
    if not os.path.abspath(file_path).startswith(os.path.abspath(web_dir)):
        return jsonify({"status": "error", "message": "Forbidden"}), 403
    
    if os.path.exists(file_path):
        return send_from_directory(web_dir, filename)
    return jsonify({"status": "error", "message": "File not found"}), 404



@app.errorhandler(404)
def not_found(error) -> Tuple[Dict[str, str], int]:
    """Handle 404 errors."""
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404



@app.errorhandler(500)
def internal_error(error) -> Tuple[Dict[str, str], int]:
    """Handle 500 errors."""
    return jsonify({"status": "error", "message": "Internal server error"}), 500


if __name__ == "__main__":
    import sys
    from src.config import Config
    
    config = Config()
    log_level = logging.INFO
    if config.verbose:
        log_level = logging.DEBUG
    
    logging.basicConfig(level=log_level)
    
    port = getattr(config, "api_port", 5000)
    debug = getattr(config, "verbose", False)
    
    logger.info(f"Starting API server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
