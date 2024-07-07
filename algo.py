from .client import Client
import time
import numpy as np
import asyncio
import sys

class Team_6_Bot(Client):
    def __init__(self):
        super().__init__()
        self.name = "Theomers"  # Your Bot's Name
        # self.auctions_participating = set()
        self.last_bids = {}
        self.timers = {}
        self.wait_time = 5
        # Your Initialization Code Here

    async def start(self, auction_id):
        await super().start(auction_id)
        # self.auctions_participating.add(auction_id)
        # bid = np.random.uniform(0.002, 0.02)
        await asyncio.sleep(5+2*np.random.randn())
        bid = 0.01
        self.last_bids[auction_id] = bid
        self.timers[auction_id] = time.time()
        await super().submit_bid(auction_id, bid)
        # Your code for starting an auction

    async def receive_bid(self, auction_id, bid_value):
        # if auction_id not in self.last_bids:
        #     self.last_bids[auction_id] = 0
        
        await super().receive_bid(auction_id, bid_value)
        time_passed = time.time() - self.timers[auction_id]
        # self.timers[auction_id] = time.time()

        # if time_passed < 30:
        # x = self.wait_time-time_passed-0.8
        # await asyncio.sleep(x if x > 0 else self.wait_time)
        # if self.wait_time >= 2.5:
        #     self.wait_time /= 2
        # bid = max(bid_value + np.random.uniform(sys.float_info.epsilon, 0.02), self.last_bids[auction_id]*1.125)

        await asyncio.sleep(5+2*np.random.randn())
        bid = max(self.last_bids[auction_id]*1.2, bid_value + 0.01)
        await super().submit_bid(auction_id, bid)
        
        # await super().submit_bid(auction_id, self.last_bids[auction_id]*1.2)
        

        # self.last_bids[auction_id] = max(bid_value + sys.float_info.epsilon, self.last_bids[auction_id]*1.125)
        self.last_bids[auction_id] = bid

        self.timers[auction_id] = time.time()

        # Your code for receiving bids

    async def end_auction(self, auction_id):
        await super().end_auction(auction_id)
        # Your code for ending auction
