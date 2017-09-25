package umich.ML.mcwrap;

import org.lwjgl.opengl.GL11;

import net.minecraft.client.renderer.entity.*;
import net.minecraft.client.model.*;
import net.minecraft.entity.EntityLivingBase;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;

@SideOnly(Side.CLIENT)
public class RenderCowCustom extends RenderCow {
	
	private float scale;
	public RenderCowCustom(RenderManager p_i46127_1_, ModelBase model, float shadow, float scale) {
		super(p_i46127_1_, model, shadow);
		this.scale = scale;
	}

	@Override
	protected void preRenderCallback(EntityLivingBase entity, float arg) {
		this.scale(entity, arg);
	}
	
	protected void scale(EntityLivingBase entity, float arg) {
		GL11.glScalef(this.scale, this.scale, this.scale);
	}
}