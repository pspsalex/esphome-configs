import esphome.codegen as cg
import esphome.config_validation as cv

# Define the component namespace
timegm_helper_ns = cg.esphome_ns.namespace("timegm_helper")

# Define the component class
TimegmHelperComponent = timegm_helper_ns.class_("TimegmHelperComponent", cg.Component)

# Configuration schema (empty since we don't need any configuration)
CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(TimegmHelperComponent),
}).extend(cv.COMPONENT_SCHEMA)

async def to_code(config):
    # Create the component instance
    var = cg.new_Pvariable(config[cv.CONF_ID])
    await cg.register_component(var, config)
