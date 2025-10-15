# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Created by: tpetit@pollitics.com
# www.pollitics.com

import generate_population

def run():
    JSON_DIR = "json_communes"
    SOURCE_DIR = "source_communes"
    BATCH_SIZE = 5
    TOTAL_BATCHES = 60
    MAX_TIME_SECONDS = 30
    generate_population.generate(JSON_DIR,SOURCE_DIR,BATCH_SIZE,TOTAL_BATCHES,MAX_TIME_SECONDS)

if __name__ == "__main__":
    run()


