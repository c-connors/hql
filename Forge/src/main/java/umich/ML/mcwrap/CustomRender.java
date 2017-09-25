package umich.ML.mcwrap;
import net.minecraftforge.fml.client.registry.RenderingRegistry;
import net.minecraft.entity.passive.*;
import net.minecraft.client.Minecraft;
import net.minecraft.client.model.*;
import net.minecraft.entity.monster.*;

public class CustomRender {
	public static void register() {
		RenderingRegistry.registerEntityRenderingHandler(EntityZombie.class, 
					new RenderZombieCustom(Minecraft.getMinecraft().getRenderManager(), 0.9F));
		RenderingRegistry.registerEntityRenderingHandler(EntityCreeper.class, 
				new RenderCreeperCustom(Minecraft.getMinecraft().getRenderManager(), 0.8F));
		RenderingRegistry.registerEntityRenderingHandler(EntityChicken.class, 
				new RenderChickenCustom(Minecraft.getMinecraft().getRenderManager(), new ModelChicken(), 0.0F, 1.2F));
		RenderingRegistry.registerEntityRenderingHandler(EntityCow.class, 
				new RenderCowCustom(Minecraft.getMinecraft().getRenderManager(), new ModelCow(), 0.0F, 0.7F));
		RenderingRegistry.registerEntityRenderingHandler(EntityPig.class, 
				new RenderPigCustom(Minecraft.getMinecraft().getRenderManager(), new ModelPig(), 0.0F, 0.8F));
		RenderingRegistry.registerEntityRenderingHandler(EntityHorse.class, 
				new RenderHorseCustom(Minecraft.getMinecraft().getRenderManager(), new ModelHorse(), 0.0F, 0.7F));
		RenderingRegistry.registerEntityRenderingHandler(EntityOcelot.class, 
				new RenderOcelotCustom(Minecraft.getMinecraft().getRenderManager(), new ModelOcelot(), 0.0F, 1.3F));
		RenderingRegistry.registerEntityRenderingHandler(EntitySheep.class, 
				new RenderSheepCustom(Minecraft.getMinecraft().getRenderManager(), new ModelSheep2(), 0.0F, 0.8F));
	}
}