@"

&#x20; # Flow-Aware Music Transition System



&#x20; This project builds a flow-aware music recommendation pipeline focused on preserving listener state across song

&#x20; transitions.



&#x20; ## Current Status



&#x20; Phase 1 is complete.



&#x20; Implemented:

&#x20; - Spotify playlist-based dataset construction

&#x20; - cleaned song-level dataset

&#x20; - positive and negative transition pair generation

&#x20; - Phase 1 weighted global feature compatibility model

&#x20; - baseline, weighted, and hard-negative experiments



&#x20; ## Project Goal



&#x20; The system is designed to generate queues that preserve musical continuity instead of optimizing only for discovery or

&#x20; engagement.



&#x20; ## Planned Pipeline



&#x20; 1. Phase 1: Weighted global audio feature similarity

&#x20; 2. Phase 2: Siamese envelope matching using Spotify audio analysis

&#x20; 3. Phase 3: Boundary transition classifier

&#x20; 4. Anchor drift guard

&#x20; 5. Beam search queue generation



&#x20; ## Environment Variables



&#x20; Create a `.env` file using `.env.example` as reference.



