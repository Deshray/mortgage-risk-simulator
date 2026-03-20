"""
train.py — Train and save the mortgage default model.

Run: python train.py
"""
import logging
from core.data import generate_portfolio, PortfolioConfig
from core.models import train, save_bundle

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Generating synthetic Canadian mortgage portfolio...")
    cfg = PortfolioConfig(n_borrowers=10_000)
    df = generate_portfolio(cfg)
    logger.info(f"Portfolio: {len(df):,} borrowers | Default rate: {df['defaulted'].mean():.3%}")

    logger.info("Training models...")
    bundle = train(df)
    save_bundle(bundle)

    logger.info("Training complete. Metrics:")
    for k, v in bundle.metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")