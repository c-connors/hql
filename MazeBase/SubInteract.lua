local SubInteract, parent = torch.class('SubInteract', 'SubTask')

function SubInteract:__init(vocab, param, maze)
    parent.__init(self, vocab, param, maze)
    self.name = param.name
    self.interact_type = g_objects[g_obj_idx[self.name]].interact
    assert(self.name)
    assert(self.vocab[self.name], self.name .. " does not exist in vocab!")
    assert(self.interact_type) 
end

function SubInteract:object_need()
    return 1
end

function SubInteract:add_necessary_item(beginning, add_item)
    local items = self.maze.items_byname[self.name]
    if add_item or items == nil or #items == 0 then
        local attr = {type = "object"}
        merge_table(attr, g_objects[g_obj_idx[self.name]])
        self.maze:place_item_rand(attr)
    end
end

function SubInteract:update()
end

function SubInteract:get_reward(is_last)
    if self.finished then
        return -self.finish_cost, false
    else
        return 0, false
    end
end

function SubInteract:is_success()
    return self.finished
end

function SubInteract:is_active()
    return not self.finished
end

function SubInteract:instruction(sentence)
    sentence:fill(self.vocab["nil"])
    sentence[1][1] = self.vocab["interact"]
    sentence[1][2] = self.vocab["with"]
    sentence[1][3] = self.vocab[self.name]
end

function SubInteract:on_item_pickup(item)
    if not self.finished then
        if item.name == self.name and self.interact_type == "pickup" then
            self.finished = true
        end
    end
    return self.finished
end

function SubInteract:on_item_broken(item)
    if not self.finished then
        if item.name == self.name and self.interact_type == "transform" then
            self.finished = true
        end
    end
    return self.finished
end
