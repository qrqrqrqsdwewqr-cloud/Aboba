from ui_controller import UIController, TemplateMatch

class FakeUI(UIController):
    def __init__(self, states):
        super().__init__(); self.states = states.copy(); self.clicks = []
    def _category_matches(self, category):
        state = self.states.get(category)
        if state is None:
            return TemplateMatch(False,0.0), TemplateMatch(False,0.0)
        return (TemplateMatch(state,0.9 if state else 0.1,10,10,1.0), TemplateMatch(not state,0.9 if not state else 0.1,10,10,1.0))
    def set_category_state(self, category, checked):
        state = self.get_category_state(category)
        if state is None: return False
        if state != checked:
            self.clicks.append(category); self.states[category] = checked
        return True

def test_select_categories_exact_1_4():
    ui = FakeUI({1:False,2:True,3:False,4:False})
    assert ui.select_categories([1,4])
    assert ui.states == {1:True,2:False,3:False,4:True}
    assert ui.clicks == [1,2,4]

def test_unknown_checkbox_blocks_submit_path():
    ui = FakeUI({1:None,2:False,3:False,4:False})
    assert not ui.select_categories([1])

def test_checked_category_not_clicked_again():
    ui = FakeUI({1:True,2:False,3:False,4:False})
    assert ui.select_categories([1])
    assert ui.clicks == []
