import random
import sys
import time
import math

from sample_players import DataPlayer
from collections import defaultdict
from isolation import isolation
from copy import deepcopy


import logging
logging.basicConfig(level=logging.WARN)


class MCTSNode():
    """
    Monte Carlo Tree Search node class
    """

    def __init__(self, state: isolation, action=None, parent=None):
        '''
        @param state: Game state included in this node
        @param parent: Parent node for current node
        '''

        self.state = state
        self.parent = parent
        self.children = []
        # The action that led to this state from parent.
        # Default is None (as in root node)
        self.action = action
        self.visit = 0
        self.utility = 0
        self.untried_actions = state.actions()


    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=0.5):
        """
        Returns the state resulting from taking the best action.
        c_param value between 0 (max score) and 1 (prioritize exploration)
        """
        maxscore = -999
        maxactions = []
        for t in self.children:
            score = (t.utility / t.visit +
                     c_param * math.sqrt(2 * math.log(self.visit / t.visit)))
            if score > maxscore:
                maxscore = score
                del maxactions[:]
                maxactions.append(t)
            elif score == maxscore:
                maxactions.append(t)

        # Choose move that results in giving us more freedom
        helper_scores = [self.heuristic_score(x.state) for x in maxactions]
        return maxactions[helper_scores.index(max(helper_scores))]

    def heuristic_score(self, state):
        # A list containing the position of open liberties in the
        # neighborhood of the starting position
        player_id = state.player()
        own_loc = state.locs[player_id]
        opp_loc = state.locs[1 - player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

    def rollout_policy(self, possible_moves):
        return random.choice(possible_moves)

    def expand(self):
        """
        Returns a state resulting from taking an action from
        the list of untried nodes
        """
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        next_state = self.state.result(action)
        child_node = MCTSNode(next_state, action=action, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.terminal_test()

    def playout_policy(self, player_id):
        """
        The simulation to run when visiting unexplored nodes.
        Defaults to uniform random moves
        """
        rollout_state = self.state
        while not rollout_state.terminal_test():
            rollout_state = rollout_state.result(
                self.rollout_policy(rollout_state.actions()))
        # Utility with respect to the player of the parent state
        if rollout_state.utility(
                1-rollout_state.player()) == float('inf'):
            return 1
        # Utility with respect to the player of the parent state
        elif rollout_state.utility(
                1-rollout_state.player()) == float('-inf'):
            return -1
        else:
            return 0

    def backpropagate(self, result):
        self.visit += 1
        self.utility += result
        result = -result
        if self.parent:
            self.parent.backpropagate(result)


class MCTSSearch():
    '''
    Perform Monte Carlo Tree Search
    '''

    def __init__(self, node: MCTSNode):
        self.root = node
        self.player_id = self.root.state.player()
        self.node_no = 1

    def best_action(self, simulations_time, c_param):
        start = time.time()
        allowed_time = math.ceil(simulations_time * 0.75)
        while (time.time()-start)*1000 <= allowed_time:
            v = self.tree_policy()
            reward = v.playout_policy(self.player_id)
            v.backpropagate(reward)
        next_node = self.root.best_child(c_param=c_param)
        next_action = next_node.action
        return next_action

    def tree_policy(self):
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                new_node = current_node.expand()
                self.node_no += 1
                return new_node
            else:
                current_node = current_node.best_child(c_param=0.5)
        return current_node


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE:
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #         call self.queue.put(ACTION) at least once before time expires
        #         (the timer is automatically managed for you)

        try:
            if state.terminal_test() or state.ply_count < 2:
                self.queue.put(random.choice(state.actions()))
            else:
                # self.queue.put(random.choice(state.actions()))
                mcts = MCTSSearch(MCTSNode(state))
                next_action = mcts.best_action(150, 0.5)
                if next_action:
                    self.queue.put(next_action)
                elif state.actions():
                    self.queue.put(random.choice(state.actions()))
                else:
                    self.queue.put(None)

        except Exception as e:
            print("Unexpected error: {}".format(sys.exc_info()[0]))
