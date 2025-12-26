import init, { solve } from '../pkg/dimension.js';

async function run() {
    await init();

    const solveBtn = document.getElementById('solve-btn');
    const input = document.getElementById('constraints');
    const resultsSection = document.getElementById('results-section');
    const scoreValue = document.getElementById('score-value');
    const bonusValue = document.getElementById('bonus-value');
    const pyramid = document.getElementById('pyramid');
    const failureMessage = document.getElementById('error-message');

    solveBtn.addEventListener('click', () => {
        const constraints = input.value.trim();
        if (!constraints) return;

        try {
            const result = solve(constraints);

            if (result.error) {
                showError(result.error);
                return;
            }

            displayResult(result);
        } catch (e) {
            showError("An unexpected error occurred: " + e);
        }
    });

    // Also solve on Enter key
    input.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') {
            solveBtn.click();
        }
    });

    function showError(msg) {
        resultsSection.style.display = 'block';
        failureMessage.textContent = msg;
        failureMessage.style.display = 'block';

        // Hide success elements
        scoreValue.parentElement.style.display = 'none';
        bonusValue.parentElement.style.display = 'none';
        pyramid.style.display = 'none';
    }

    function displayResult(result) {
        resultsSection.style.display = 'block';
        failureMessage.style.display = 'none';

        // Show success elements
        scoreValue.parentElement.style.display = 'block';
        bonusValue.parentElement.style.display = 'block';
        pyramid.style.display = 'flex'; // grid/flex?

        scoreValue.innerText = result.score;
        bonusValue.innerText = result.flag ? "Yes (+2 pts)" : "No"; // Assuming Bonus means Flag

        renderPyramid(result.positions);
    }

    function renderPyramid(positions) {
        pyramid.innerHTML = '';

        // Positions map to indices 0..10.
        // Based on lib.rs:
        // L0 (top): 0
        // L1 (middle): 1, 2, 3
        // L2 (base): 4, 5, 6, 7, 8, 9, 10 -- wait, need to check graph in lib.rs

        /*
        // L0      L1                L2
        //   1                1
        //  /|\              /
        // 6 | 2   6--9-----7--2       9    7
        // |\|/|     / \   /            \  /
        // | 0 |    5   \ /              10
        // |/|\|         8--3             |
        // 5 | 3          \               8
        //  \|/            4
        //   4
        
        Actually, let's look at lib.rs again.
        It seems 0 is "middle"?
        The graph says:
        L0: ?
        
        Wait, the comment in lib.rs says:
        // Describe the relationship of each position on the board to all other positions.
        // L0      L1                L2
        
        Actually position 0 seems deeply connected.
        Usually Pyramid is:
        Top: 1 sphere
        Middle: 3 spheres
        Bottom: 7 spheres? Or 6?
        Total 11 spheres.
        Standard pyramid (tetrahedron) for Dimension game is usually:
        Base: Hexagon of 7 spheres? Or Triangle?
        The game "Dimension" by Kosmos has 15 spheres? Stacked?
        Wait, the code says `NUM_POSITIONS = 11`.
        
        Let's assume the positions are layers.
        If I look at `below` mask in `PositionNode`:
        Pos 7 below: 0b00000000111 -> 0, 1, 2 ?
        
        Let's check `lib.rs` quickly to understand visual layout.
        The goal is just to show them.
        If I can just put them in a list or some structure.
        
        For now, I will render them in a simple layout, maybe just a list of spheres if I don't know the geometry,
        but I can probably deduce it.
        
        If `below` is non-zero, it is ON TOP of something.
        Pos 7 has below: 0, 1, 2. So 7 is above 0, 1, 2.
        Pos 8 below: 0, 3, 4? Mask 0b00000011001 -> 0, 3, 4.
        Pos 9 below: 0, 1, 5?
        Pos 10 below: 0, ?
        
        It seems L2 are the ones ON TOP.
        L0/L1 are base?
        
        Actually `PositionNode` `below` field means "positions below this one".
        If `below` is 0, it is on the table (base).
        Base positions: 0, 1, 2, 3, 4, 5, 6.
        Positions with non-zero below: 7, 8, 9, 10.
        
        Wait, 7, 8, 9, 10 are on top of others.
        And 10 seems to be on top of others too?
        Pos 10 below mask: 0b01110000000 -> 8, 9, 10? No.
        Mask is 11 bits.
        Let's re-read `lib.rs`.
        Pos 10 (last index): 
           below: 0b01110000000 => 7, 8, 9 are below 10?
           Wait, bit 0 is Pos 0.
           0b01110000000 => Bits 8, 9, 10 are set?
           (1<<8) | (1<<9) | (1<<10)?
           If Pos 10 is the index, mask `below` describes who supports it.
           If bits 8, 9 are set, then 8 and 9 support 10.
        
        So:
        Level 1 (Base): 0..6 (7 spheres).
        Level 2: 7, 8, 9 (3 spheres).
        Level 3: 10 (1 sphere).
        
        Total 7 + 3 + 1 = 11.
        Yes.
        
        So I will render 3 levels.
        Level 1 (Bottom): indices 0, 1, 2, 3, 4, 5, 6
        Level 2 (Middle): indices 7, 8, 9
        Level 3 (Top): index 10
        
        Visual stack:
           [10]
          [7][8][9]
        [0][1][2][3][4][5][6]
        
        I will create 3 container divs in JS.
        */

        const levels = [
            { name: 'level-0', indices: [10] }, // Top
            { name: 'level-1', indices: [7, 8, 9] },
            { name: 'level-2', indices: [0, 1, 2, 3, 4, 5, 6] } // Base from visual perspective
        ];

        // We want to physically stack them BOTTOM UP in HTML if we use flex/grid?
        // Actually visual order top to bottom is L3 -> L2 -> L1.

        levels.forEach(level => {
            const row = document.createElement('div');
            row.className = level.name;
            // Just center them
            row.style.display = 'flex';
            row.style.justifyContent = 'center';
            row.style.marginTop = level.name === 'level-0' ? '0' : '-15px'; // Overlap

            level.indices.forEach(idx => {
                const colorChar = positions[idx]; // "K", "W", "B", etc. or null
                const sphere = document.createElement('div');
                sphere.className = 'sphere';
                if (colorChar) {
                    sphere.classList.add(colorChar);
                } else {
                    // Empty position?
                    sphere.style.opacity = '0.2';
                    sphere.style.backgroundColor = '#333';
                }
                // Add tooltip or text?
                row.appendChild(sphere);
            });
            pyramid.appendChild(row);
        });

        // Setup flex column for pyramid container
        pyramid.style.display = 'flex';
        pyramid.style.flexDirection = 'column';
        pyramid.style.alignItems = 'center';
    }
}

run();
