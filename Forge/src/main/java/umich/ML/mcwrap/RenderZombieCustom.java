package umich.ML.mcwrap;

import org.lwjgl.opengl.GL11;

import net.minecraft.client.renderer.entity.RenderManager;
import net.minecraft.client.renderer.entity.RenderZombie;
import net.minecraft.entity.EntityLivingBase;
import net.minecraftforge.fml.relauncher.Side;
import net.minecraftforge.fml.relauncher.SideOnly;

@SideOnly(Side.CLIENT)
public class RenderZombieCustom extends RenderZombie {
	
	private float scale;
	public RenderZombieCustom(RenderManager p_i46127_1_, float scale) {
		super(p_i46127_1_);
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