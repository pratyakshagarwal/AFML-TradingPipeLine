import os, orjson, websockets, asyncio, logging, datetime as dt
from tqdm import tqdm
from typing import AsyncGenerator, Dict, Any

class LimitOrderSnapshot:
    def __init__(self, data):
        self.type = "depth"
        self.event_time = data.get("E")
        self.bids = [[float(p), float(q)] for p, q in data.get("b", [])]
        self.asks = [[float(p), float(q)] for p, q in data.get("a", [])]

    @property
    def data(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "event_time": self.event_time,
            "bids": self.bids,
            "asks": self.asks
        }

class TradeSnapshot:
    def __init__(self, data: Dict[str, Any]):
        self.type = "trade"
        self.event_time = data.get("E")                   # Event time (Binance server time)
        self.trade_id = data.get("t")                     # Unique trade ID
        self.price = float(data.get("p", 0.0))            # Trade price
        self.quantity = float(data.get("q", 0.0))         # Trade quantity
        self.buyer_order_id = data.get("b")               # Optional: buyer order ID
        self.seller_order_id = data.get("a")              # Optional: seller order ID
        self.trade_time = data.get("T")                   # Trade execution time
        self.is_buyer_maker = bool(data.get("m", False))  # True if buyer is maker (passive side)

    @property
    def data(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "event_time": self.event_time,
            "trade_id": self.trade_id,
            "price": self.price,
            "quantity": self.quantity,
            "buyer_order_id": self.buyer_order_id,
            "seller_order_id": self.seller_order_id,
            "trade_time": self.trade_time,
            "is_buyer_maker": self.is_buyer_maker
        }


class DataFetcher:
    def __init__(self, 
                 asset: str, 
                 interval_ms: int = 100,
                 type: str = "trade"):
        
        self.asset, self.type  = asset.lower(), type
        self.save_dir = os.path.join("data", self.asset)
        os.makedirs(self.save_dir, exist_ok=True)

        if type.lower()=="limit_order": 
            self.dType = LimitOrderSnapshot
            self.depth_stream = f"wss://stream.binance.com:9443/ws/{self.asset}@depth@{interval_ms}ms"
        elif type.lower()=="trade":
            self.dType = TradeSnapshot
            self.depth_stream = f"wss://stream.binance.com:9443/ws/{self.asset}@trade"
        else: raise ValueError("type must be 'limit_order' or 'trade'")

        self.counter = 0
        self.progress_bar = None

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    async def stream(self, reconnects: int = 3) -> AsyncGenerator[Dict[str, Any], None]:
        for attempt in range(1, reconnects + 1):
            try:
                async with websockets.connect(self.depth_stream, ping_interval=None) as ws:
                    self.logger.info(f"Connected to {self.depth_stream}")
                    
                    async for message in ws:
                        data = orjson.loads(message)
                        parsed = self.dType(data).data
                        yield parsed
                        
            except (websockets.ConnectionClosed, asyncio.TimeoutError) as e:
                self.logger.warning(f"Connection lost (attempt {attempt}/{reconnects}): {e}")
                if attempt < reconnects:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"Error during stream: {e}", exc_info=True)
                break

    async def fetch(self, reconnects: int = 3, nRows:int = 1000):
        for attempt in range(1, reconnects + 1):
            try:
                async with websockets.connect(self.depth_stream, ping_interval=None) as ws:
                    self.logger.info(f"Connected to {self.depth_stream}")
                    self.progress_bar = tqdm(total=nRows, desc=f"Fetching {self.asset}", unit="rows")
                    
                    async for message in ws:
                        data = orjson.loads(message)
                        parsed = self.dType(data).data
                        # Save immediately
                        self._save_single(parsed)
                        
                        self.counter += 1
                        self.progress_bar.update(1)

                        if self.counter % 100 == 0:
                            self.progress_bar.set_postfix({
                                "saved": f"{self.counter}/{nRows}",
                                "asset": self.asset
                            })

                        if self.counter >= nRows:
                            self.logger.info(f"Saved {self.counter} rows to {self.save_dir}")
                            self.progress_bar.close()
                            return
                    
                    break

            except (websockets.ConnectionClosed, asyncio.TimeoutError) as e:
                self.logger.warning(f"Connection lost (attempt {attempt}/{reconnects}): {e}")
                if attempt < reconnects:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                self.logger.error(f"Error during fetch: {e}", exc_info=True)
                break
        
        if self.progress_bar:
            self.progress_bar.close()

    def _get_filename(self):
        """Generate filename based on current hour."""
        now = dt.datetime.now().strftime("%Y%m%d%H")
        return os.path.join(self.save_dir, f"{now}_{self.type}data.jsonl")

    def _save_single(self, data: dict):
        """Save a single row immediately to file."""
        filename = self._get_filename()
        with open(filename, "a") as f:
            f.write(orjson.dumps(data).decode() + "\n")
            
if __name__ == "__main__":
    async def main():
        fetcher = DataFetcher(asset="btcusdt", type="trade", interval_ms=1000)
        # async for batch in fetcher.stream():
        #     print(batch)
        await fetcher.fetch(nRows=500000)
asyncio.run(main())
