
function processInputsIntoLoadsOfPatterns(N){
    for (int y = 0; y <= imageHeight - N; y++) {
        for (int x = 0; x <= imageWidth - N; x++) {
            patterns.add(getImageArea(x, y, N, N), "0deg"); // Add the subimage at (x, y), size (N, N)
            patterns.add(getImageArea(x, y, N, N), "90deg");
            patterns.add(getImageArea(x, y, N, N), "180deg");
            patterns.add(getImageArea(x, y, N, N), "270deg");
        }
    }
    for (int i = 0; i < patterns.size(); i++) {
        for (int j = i + 1; j < patterns.size(); j++) {
            if (patterns[i] == patterns[j]) { // If the pattern is identical
                patterns[i].frequency++; // Count up the frequency
                patterns.remove(patterns[j]); // Delete the identical twin
                j--; // Go back one, so we don't skip the index
                continue;
            }
        }
    }
}

function initializeOutput(){
    for (int y = 0; y <= outputHeight; y++) {
        for (int x = 0; x <= outputWidth; x++) {
            wave.at(x, y) = allPossiblePatterns;
        }
    }
}

function findLowestEntropyCells() {
    int currentLowestEntropy = -1; // Undefined entropy at first
    Cell[] cells = []; // Don't care, this is not actual code anyway
    for (int y = 0; y <= outputHeight; y++) {
        for (int x = 0; x <= outputWidth; x++) {
            if (wave.at(x, y).isDefinite()) {
                continue; // We won't want to deal with definite cells; we have observed them already
            }
            if (currentLowestEntropy == -1 ||  // If the lowest is still undefined or
                wave.at(x, y).entropy < currenetLowestEntropy) { // If this one is lower than the current lowest entropy
                cells = []; // we clear the cells
                currentLowestEntropy = wave.at(x, y).entropy;
            }
            if (currentLowestEntropy == wave.at(x, y).entropy) { // This, together with the if above, 
                                                                 // means if this cell's entropy is less or equal than the current lowest entropy, we add it to the cells array.
                cells.push(wave.at(x, y));
            }
        }
    }
    return cells;
}


function observe(){

    
    cells = findLowestEntropyCells();
    if (cells.size() == 0) {
        return DONE;
    } else if (cells[0].entropy == 0) {
        return CONTRADICTION;
    } else {
        // We're fine
    }

    cells.chooseOneAtRandom().collapse();
    flag(cells.neighbors());
    Cell::collapse() {
        int e = distribute(0, entropy);
        int i;
        for (i = 0; i < patterns.size(); i++) {
            e -= patterns[i].frequency;
            if (e <= 0) {
                break;
            }
        }
        patterns.splice(0, i); // or patterns.erase(patterns.begin(), patterns.begin() + i)
        patterns.splice(1, this.patterns.length - 1); // or patterns.erase(patterns.begin() + i, patterns.end())
    }
}

function propagate(){
    while (!flagged.empty()) {
        Cell cell = flagged.pop();
        bool dirty = false; // At first we assume no patterns of this cell has been collapsed
        for (int i = 0; i < cell.patterns.size(); i++) {
            bool legit = true; // At first we assume this pattern of this cell is legit
            for (int j = 0; j < cell.patterns[i].rules.size(); j++) {
                Vec2 offset = cell.patterns[i].rules[j].offset;
                Cell anotherCell = cell.position + offset;
                if (anotherCell.violates(cell.patterns[i].rules[j])) {
                    // This pattern could not be placed here anymore.
                    legit = false;
                    break;
                }
            }
            if (!legit) { // If not legit anymore
                cell.patterns.splice(i, 1);
                dirty = true;
                i--;
            }
        }
    
        if (dirty) {
            flag(cell.neighbors()); // Yep
        }
    }
}

int main() {
    getInput();
    processInputsIntoLoadsOfPatterns(N);
    initializeOutput();
    while (notFullyCollapsed()) {
        observe();
        propagate();
    } 
}