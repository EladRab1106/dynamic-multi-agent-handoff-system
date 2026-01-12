import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "agents.direct_answer.service:app",
        host="0.0.0.0",
        port=8003,
        reload=True, 
    )
